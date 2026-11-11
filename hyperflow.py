"""HyperFlow H3 Flash: load its GGUF and INT8 safetensors checkpoints, and the step schedule they sample.

HyperFlow H3 Flash is MiniMax H3 distilled to a few steps. Some of its blocks are pruned, some
linears are factorized into two low-rank halves, and it takes a second time input: the endpoint r
of each step beside its time t. The loader rebuilds that on ComfyUI's own MiniMax H3 model.

- Factorized linears `<name>.down` / `<name>.up` replace the dense `attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`
  and `mlp.fc2` of the blocks that store them; split `q/k/v_proj` are concatenated back into one projection.
  Dense linears, quantized or not, stay with ComfyUI's loader.
- Two-time checkpoints embed the step endpoint r next to t: temb = te(t) + gate * (ee(r) - te(t)).
  Text and video use (1 - sigma, 1 - sigma_next); audio uses the same pair mapped to its own shift; condition
  rows keep r = t. sigma_next comes from the running schedule in `transformer_options["sample_sigmas"]`.
- A GGUF file holds GGML-quantized tensors, which this pack's ggml_ops dequantize on the fly. An INT8 convrot
  safetensors file stores every INT8 linear as a ComfyUI `int8_tensorwise` layer (`weight`, `weight_scale`,
  `comfy_quant`), so ComfyUI's loader and kernels run them as is; its low-rank factors stay bf16. The
  `hyperflow` metadata entry marks those files and carries the layout version in `format`.

Adapted from ComfyUI-HyperFlow-Flash by Video Rebirth
(https://github.com/Video-Rebirth/ComfyUI-HyperFlow-Flash, version 1.1.0, commit c807a89), under the
GNU General Public License v3.0. See NOTICE. Changed for H3 Editor on 2026-11-08 and 2026-11-09: its
gguf_io.py, safetensors_io.py, model_patch.py and loader_nodes.py are joined in this module; GGUF
tensors and layers come from this pack's ggml_ops and ggml_dequant, instead of a ComfyUI-GGUF install
found at run time, and are imported only when a .gguf is read; the folder of .gguf models is
registered here; the nodes are H3 HyperFlow Loader and H3 HyperFlow Sigmas in nodes.py.
"""

from __future__ import annotations

import contextlib
import functools
import importlib.util
import json
import logging
import os
import re
import struct
import warnings
from collections import defaultdict
from collections.abc import Callable
from types import SimpleNamespace

import comfy.ldm.minimax.model as packed_dit
import comfy.sd
import comfy.utils
import folder_paths
import torch
from torch import nn

# The .gguf files in the diffusion model folders, listed under the key ComfyUI-GGUF uses too, so
# either pack can register it first and both see the same files.
GGUF_FOLDER = "unet_gguf"
MODEL_FOLDERS = (GGUF_FOLDER, "diffusion_models", "unet")


def register_gguf_folder() -> None:
    if GGUF_FOLDER in folder_paths.folder_names_and_paths:
        return
    for key in ("diffusion_models", "unet"):
        if key in folder_paths.folder_names_and_paths:
            folder_paths.folder_names_and_paths[GGUF_FOLDER] = (folder_paths.folder_names_and_paths[key][0], {".gguf"})
            return


register_gguf_folder()

# Reading a HyperFlow GGUF.
#
# The files carry no `general.architecture` key (the layout stable-diffusion.cpp reads), so the
# tensors are read here and handed to the on-the-fly dequantizing ops. Only this path needs the
# gguf package, so it is imported when a .gguf is first read.

TENSOR_PREFIX = "model.diffusion_model."
GGUF_MISSING = (
    "HyperFlow H3 Flash GGUF files need the gguf Python package. Install this pack's requirements: "
    "pip install -r requirements.txt in its folder, then restart ComfyUI. "
    "The INT8 safetensors files load without it."
)


def gguf_missing() -> str | None:
    """Why a .gguf cannot load here, or None when it can."""
    return GGUF_MISSING if importlib.util.find_spec("gguf") is None else None


@functools.cache
def _ggml() -> SimpleNamespace:
    """The gguf package with this pack's GGUF tensors and layers."""
    try:
        import gguf
    except ImportError as error:
        raise RuntimeError(GGUF_MISSING) from error
    if __package__:
        from . import ggml_dequant, ggml_ops
    else:  # imported on its own, as the tests do
        import ggml_dequant
        import ggml_ops
    return SimpleNamespace(gguf=gguf, dequant=ggml_dequant, ops=ggml_ops)


def operations():
    ops = _ggml().ops.GGMLOps()
    ops.Linear.dequant_dtype = None
    ops.Linear.patch_dtype = None
    return ops


def gguf_placeholder(like: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """A zero-byte GGUF weight with the dense shape; the module holding it is replaced before any forward."""
    return _ggml().ops.GGMLTensor(
        torch.empty(0, dtype=torch.uint8), tensor_type=like.tensor_type, tensor_shape=torch.Size(shape)
    )


def to_float(tensor: torch.Tensor) -> torch.Tensor:
    return _ggml().dequant.dequantize_tensor(tensor, dtype=torch.float32).reshape(tensor.tensor_shape)


def read(path: str) -> dict[str, torch.Tensor]:
    ggml = _ggml()
    types = ggml.gguf.GGMLQuantizationType
    reader = ggml.gguf.GGUFReader(path)
    state_dict = {}
    for tensor in reader.tensors:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            data = torch.from_numpy(tensor.data)
        shape = torch.Size(int(size) for size in reversed(tensor.shape))
        if tensor.tensor_type in (types.F32, types.F16):
            data = data.view(*shape)
        value = ggml.ops.GGMLTensor(data, tensor_type=tensor.tensor_type, tensor_shape=shape)
        if len(shape) <= 1 and tensor.tensor_type == types.BF16:
            value = ggml.dequant.dequantize_tensor(value, dtype=torch.float32)
        state_dict[tensor.name.removeprefix(TENSOR_PREFIX)] = value
    quantized = [name for name, value in state_dict.items() if ggml.dequant.is_quantized(value)]
    if quantized:
        state_dict[max(quantized, key=lambda name: state_dict[name].numel())].is_largest_weight = True
    try:
        import comfy.storage
    except ImportError:
        return state_dict
    # Lets ComfyUI's storage policy (--fast-disk, NVMe detection) see which file backs these tensors.
    annotate = getattr(comfy.storage, "annotate_state_dict", None)
    if annotate is not None:
        annotate(state_dict, path)
    return state_dict


# Reading a HyperFlow safetensors file, in ComfyUI's own tensor layout.

METADATA_KEY = "hyperflow"
SUPPORTED_FORMAT = 1
MAX_HEADER_BYTES = 64 << 20
# comfy_kitchen's triton INT8 GEMM indexes with int32 and corrupts or faults once rows x width reaches 2^31.
ROW_ELEMENTS = 1 << 30


@functools.lru_cache(maxsize=512)
def _header(path: str, mtime_ns: int, size: int) -> dict | None:
    del mtime_ns, size  # cache keys: a rewritten file is read again
    try:
        with open(path, "rb") as file:
            length = struct.unpack("<Q", file.read(8))[0]
            if length > MAX_HEADER_BYTES:
                return None
            metadata = json.loads(file.read(length)).get("__metadata__") or {}
        info = json.loads(metadata[METADATA_KEY]) if METADATA_KEY in metadata else None
        return info if isinstance(info, dict) else None
    except (OSError, ValueError, TypeError, AttributeError, struct.error):
        # Any file can sit in the model folder under a .safetensors name; none of them may break the list.
        return None


def hyperflow_info(path: str) -> dict | None:
    """The `hyperflow` metadata of a safetensors file, or None for any other file."""
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return _header(path, stat.st_mtime_ns, stat.st_size)


def checked_info(path: str) -> dict:
    info = hyperflow_info(path)
    if info is None:
        raise RuntimeError(f"{path} is not a HyperFlow checkpoint (no `{METADATA_KEY}` metadata)")
    version = int(info.get("format", 0))
    if version > SUPPORTED_FORMAT:
        raise RuntimeError(
            f"{path} uses HyperFlow layout {version}; this H3 Editor reads up to {SUPPORTED_FORMAT}. Update the pack."
        )
    return info


def dense_placeholder(like: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """A zero-stride bf16 weight with the dense shape: ComfyUI keeps it without a copy at this model's bf16
    compute dtype, and the module holding it is replaced before any forward."""
    del like
    return torch.zeros((), dtype=torch.bfloat16).expand(shape)


def float32_tensors(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: tensor
        for name, tensor in state_dict.items()
        if tensor.dtype == torch.float32 and not name.endswith(".weight_scale")
    }


def keep_float32(root: nn.Module, tensors: dict[str, torch.Tensor]) -> int:
    """Restore the fp32 parameters that ComfyUI's mixed-precision ops cast to bf16 at load."""
    restored = 0
    for name, tensor in tensors.items():
        module_name, _, leaf = name.rpartition(".")
        try:
            module = root.get_submodule(module_name)
        except AttributeError:
            continue
        current = module._parameters.get(leaf)
        if current is None or current.dtype == torch.float32 or tuple(current.shape) != tuple(tensor.shape):
            continue
        module._parameters[leaf] = nn.Parameter(tensor.to(device=current.device), requires_grad=False)
        setattr(module, f"{leaf}_comfy_model_dtype", torch.float32)
        restored += 1
    return restored


def _int8(module: nn.Module | None) -> bool:
    return getattr(module, "quant_format", None) == "int8_tensorwise"


def _row_chunks(forward, width: int):
    """`forward` over the last dimension, at most `ROW_ELEMENTS // width` rows of tokens at a time."""
    rows = max(1, ROW_ELEMENTS // width)

    @functools.wraps(forward)
    def chunked(x, *args, **kwargs):
        flat = x.reshape(-1, x.shape[-1])
        if flat.shape[0] <= rows:
            return forward(x, *args, **kwargs)
        out = None
        for start in range(0, flat.shape[0], rows):
            part = forward(flat[start : start + rows], *args, **kwargs)
            if out is None:
                out = part.new_empty((flat.shape[0], part.shape[-1]))
            out[start : start + part.shape[0]] = part
        return out.reshape(*x.shape[:-1], out.shape[-1])

    return chunked


def limit_int8_rows(root: nn.Module) -> int:
    """Run token-wise INT8 work in row chunks below `ROW_ELEMENTS`: every INT8 linear, and every MLP with an INT8
    `fc2` whole, since its fused SwiGLU call bypasses `fc2.forward`."""
    wrapped = 0
    for module in root.modules():
        linear = module if _int8(module) else getattr(module, "fc2", None)
        if not _int8(linear):
            continue
        module.forward = _row_chunks(module.forward, max(linear.in_features, linear.out_features))
        wrapped += 1
    return wrapped


# HyperFlow's additions on top of ComfyUI's packed video+audio DiT.

PackedModel = packed_dit.MiniMaxH3Model
VIDEO_SHIFT_OPTION = "minimax_h3_sigma_shift_video"
AUDIO_SHIFT_OPTION = "minimax_h3_sigma_shift_audio"
ENDPOINT_PREFIX = "endpoint_time_embedder."
PART = re.compile(
    r"^blocks\.(\d+)\.(attn\.[qkv]_proj|attn\.out_proj|mlp\.fc[12])\."
    r"(down\.weight|up\.weight|weight|weight_scale|comfy_quant)$"
)
QKV_PARTS = ("attn.q_proj", "attn.k_proj", "attn.v_proj")
DENSE_LEAVES = ("weight", "weight_scale", "comfy_quant")
# (tensor of the module, dense shape) -> a zero-byte weight with that shape, replaced before any forward.
Placeholder = Callable[[torch.Tensor, tuple[int, ...]], torch.Tensor]
# Rows of the first step share t = 0 across video and audio but not r; audio moves to the next float32 value.
AUDIO_TIME_OFFSET = 1e-6


class LowRankLinear(nn.Module):
    def __init__(self, down: nn.Module, up: nn.Module) -> None:
        super().__init__()
        self.down = down
        self.up = up
        self.in_features = down.in_features
        self.out_features = up.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class SplitQKV(nn.Module):
    def __init__(self, q_proj: nn.Module, k_proj: nn.Module, v_proj: nn.Module) -> None:
        super().__init__()
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.in_features = q_proj.in_features
        self.out_features = q_proj.out_features + k_proj.out_features + v_proj.out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.q_proj(x), self.k_proj(x), self.v_proj(x)), dim=-1)


class FactorizedMLP(packed_dit.MLP):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, hidden = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(nn.functional.silu(gate) * hidden)


def _shape(tensor: torch.Tensor) -> tuple[int, ...]:
    shape = getattr(tensor, "tensor_shape", None)  # a GGUF tensor's own shape; its data may be packed
    return tuple(int(size) for size in (tensor.shape if shape is None else shape))


class FactorizedPlan:
    """Pops the factorized and split-q/k/v linears of every block and leaves dense placeholders for model detection."""

    def __init__(self, state_dict: dict[str, torch.Tensor], placeholder: Placeholder) -> None:
        found: dict[tuple[int, str], dict[str, str]] = defaultdict(dict)
        for name in state_dict:
            match = PART.match(name)
            if match is not None:
                found[int(match.group(1)), match.group(2)][match.group(3).removesuffix(".weight")] = name
        parts: dict[int, dict[str, dict[str, torch.Tensor]]] = defaultdict(dict)
        for (block, module), leaves in found.items():
            if module in QKV_PARTS or any(leaf not in DENSE_LEAVES for leaf in leaves):
                parts[block][module] = {leaf: state_dict.pop(name) for leaf, name in leaves.items()}
        self.parts = dict(parts)
        for block, modules in self.parts.items():
            for dense, shape, like in self._dense_shapes(modules):
                state_dict[f"blocks.{block}.{dense}.weight"] = placeholder(like, shape)

    @staticmethod
    def _out_in(part: dict[str, torch.Tensor]) -> tuple[int, int]:
        if "weight" in part:
            return _shape(part["weight"])
        return _shape(part["up"])[0], _shape(part["down"])[1]

    @staticmethod
    def _like(part: dict[str, torch.Tensor]) -> torch.Tensor:
        return part.get("weight", part.get("up"))

    def _dense_shapes(self, modules: dict[str, dict[str, torch.Tensor]]):
        if any(name in modules for name in QKV_PARTS):
            shapes = [self._out_in(modules[name]) for name in QKV_PARTS]
            yield "attn.qkv_proj", (sum(out for out, _ in shapes), shapes[0][1]), self._like(modules["attn.q_proj"])
        for name in ("attn.out_proj", "mlp.fc1", "mlp.fc2"):
            if name in modules:
                yield name, self._out_in(modules[name]), self._like(modules[name])

    def install(self, model: PackedModel, ops) -> int:
        for block_index, modules in self.parts.items():
            block = model.blocks[block_index]

            def build(name: str, dtype=model.dtype, modules=modules) -> nn.Module:
                part = modules[name]
                if "weight" in part:
                    out_features, in_features = _shape(part["weight"])
                    linear = ops.Linear(in_features, out_features, bias=False, dtype=dtype)
                    linear.load_state_dict(part, strict=False)
                    return linear
                rank, in_features = _shape(part["down"])
                out_features = _shape(part["up"])[0]
                down = ops.Linear(in_features, rank, bias=False, dtype=dtype)
                up = ops.Linear(rank, out_features, bias=False, dtype=dtype)
                down.load_state_dict({"weight": part["down"]}, strict=False)
                up.load_state_dict({"weight": part["up"]}, strict=False)
                return LowRankLinear(down, up)

            if any(name in modules for name in QKV_PARTS):
                block.attn.qkv_proj = SplitQKV(*(build(name) for name in QKV_PARTS))
            if "attn.out_proj" in modules:
                block.attn.out_proj = build("attn.out_proj")
            for name in ("mlp.fc1", "mlp.fc2"):
                if name in modules:
                    setattr(block.mlp, name.split(".")[1], build(name))
            if "mlp.fc2" in modules:
                block.mlp.__class__ = FactorizedMLP
        return len(self.parts)


class EndpointState:
    """Plain holder so the endpoint embedder stays registered only once, at the model root."""

    def __init__(self, embedder: nn.Module, gate: float) -> None:
        self.embedder = embedder
        self.gate = gate
        self.endpoints: dict[float, float] = {}

    def set_endpoints(self, pairs: dict[float, float]) -> None:
        keys = torch.tensor(list(pairs), dtype=torch.float32).tolist()
        self.endpoints = dict(zip(keys, pairs.values(), strict=True))

    def lookup(self, t: torch.Tensor) -> torch.Tensor:
        values = [self.endpoints.get(value, value) for value in t.float().tolist()]
        return torch.tensor(values, dtype=torch.float32, device=t.device)


class TwoTimeEmbedder(packed_dit.TimeEmbedder):
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        state: EndpointState = self.__dict__["endpoint_state"]
        current = super().forward(t)
        endpoint = state.embedder(state.lookup(t)).to(current.dtype)
        return current + state.gate * (endpoint - current)


@contextlib.contextmanager
def separated_audio_time(shift_video: float, shift_audio: float):
    original = packed_dit.time_shift_sigma

    def shifted(sigma, from_shift, to_shift):
        value = original(sigma, from_shift, to_shift)
        if (from_shift, to_shift) == (shift_video, shift_audio):
            value = value - AUDIO_TIME_OFFSET
        return value

    packed_dit.time_shift_sigma = shifted
    try:
        yield shifted
    finally:
        packed_dit.time_shift_sigma = original


def step_endpoint(sigma: torch.Tensor, sample_sigmas: torch.Tensor | None) -> torch.Tensor:
    if sample_sigmas is None:
        return sigma
    schedule = sample_sigmas.to(device=sigma.device, dtype=torch.float32)
    index = int((schedule - sigma).abs().argmin())
    return schedule[min(index + 1, schedule.shape[0] - 1)]


def _f32(value: float) -> float:
    return torch.tensor(value, dtype=torch.float32).item()


class HyperFlowModel(PackedModel):
    """Two-time forward: sets this step's (t, r) pairs on the time embedder, then runs the stock packed forward."""

    def _forward(self, x, timestep, context, transformer_options=None, **kwargs):
        transformer_options = {} if transformer_options is None else transformer_options
        state: EndpointState = self.time_embedder.__dict__["endpoint_state"]
        shift_video = float(transformer_options.get(VIDEO_SHIFT_OPTION, self.sigma_shift_video))
        shift_audio = float(transformer_options.get(AUDIO_SHIFT_OPTION, self.sigma_shift_audio))
        sigma = (timestep.flatten()[0] / 1000.0).float().clamp(min=1e-6)
        sample_sigmas = transformer_options.get("sample_sigmas")
        if sample_sigmas is None and not getattr(self, "_warned_endpoint", False):
            logging.warning("HyperFlow: no running schedule in transformer_options; using r = t")
            self._warned_endpoint = True
        sigma_next = step_endpoint(sigma, sample_sigmas)
        shift = packed_dit.time_shift_sigma
        t_video, r_video = float(1.0 - sigma), float(1.0 - sigma_next)
        r_audio = float(1.0 - shift(sigma_next, shift_video, shift_audio))
        t_audio = float(1.0 - shift(sigma, shift_video, shift_audio))
        collide = _f32(t_video) == _f32(t_audio) and r_video != r_audio
        with separated_audio_time(shift_video, shift_audio) if collide else contextlib.nullcontext() as moved:
            if moved is not None:
                t_audio = float(1.0 - moved(sigma, shift_video, shift_audio))
            state.set_endpoints({t_video: r_video, t_audio: r_audio})
            return super()._forward(x, timestep, context, transformer_options, **kwargs)


def install_two_time(model: PackedModel, endpoint_sd: dict[str, torch.Tensor], gate: float, ops) -> None:
    current = model.time_embedder
    hidden, freq_dim = _shape(endpoint_sd["proj_in.weight"])
    out_dim = _shape(endpoint_sd["proj_out.weight"])[0]
    embedder = packed_dit.TimeEmbedder(freq_dim, hidden, out_dim, dtype=torch.float32, operations=ops)
    embedder.load_state_dict(endpoint_sd, strict=False)
    model.endpoint_time_embedder = embedder
    current.__class__ = TwoTimeEmbedder
    current.__dict__["endpoint_state"] = EndpointState(embedder, gate)
    model.__class__ = HyperFlowModel


# Loading, and the 4, 6 or 8-step schedule a checkpoint samples.

# Raw interval knots by step count and the video flow shift they are mapped through.
RAW_SIGMAS = {
    4: (1.0, 0.681872, 0.205019, 0.036496, 0.0),
    6: (1.0, 0.825217, 0.471725, 0.147401, 0.049626, 0.015517, 0.0),
    8: (1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0),
}
DEFAULT_STEPS = 6
VIDEO_SHIFT = 12.0
WEIGHTS_IN = ("auto", "ram", "disk")


def step_sigmas(steps: int = DEFAULT_STEPS) -> torch.Tensor:
    raw = torch.tensor(RAW_SIGMAS[steps], dtype=torch.float64)
    return (VIDEO_SHIFT * raw / (1 + (VIDEO_SHIFT - 1) * raw)).float()


def gguf_files() -> list[str]:
    names = set()
    for key in MODEL_FOLDERS:
        if key in folder_paths.folder_names_and_paths:
            names.update(name for name in folder_paths.get_filename_list(key) if name.endswith(".gguf"))
    return sorted(names)


def safetensors_files() -> list[str]:
    """The .safetensors files that carry HyperFlow metadata; other diffusion models are left out of the list."""
    names = set()
    for key in MODEL_FOLDERS:
        if key not in folder_paths.folder_names_and_paths:
            continue
        for name in folder_paths.get_filename_list(key):
            if name.endswith(".safetensors") and name not in names:
                path = folder_paths.get_full_path(key, name)
                if path is not None and hyperflow_info(path) is not None:
                    names.add(name)
    return sorted(names)


def files() -> list[str]:
    """Every HyperFlow checkpoint in the diffusion model folders: any .gguf, and the INT8 safetensors."""
    return sorted(set(gguf_files()) | set(safetensors_files()))


def model_path(name: str) -> str:
    """The file a loader names, wherever this ComfyUI keeps it.

    A workflow made on another machine may name the file in another folder or with the other
    separator: ComfyUI lists `hyperflow/file` on Linux and macOS and `hyperflow\\file` on Windows.
    The name is tried as written, then by its file name alone among the listed files.
    """
    folders = [key for key in MODEL_FOLDERS if key in folder_paths.folder_names_and_paths]
    written = str(name or "").replace("\\", "/")
    for candidate in dict.fromkeys((name, written)):
        for key in folders:
            path = folder_paths.get_full_path(key, candidate) if candidate else None
            if path is not None:
                return path
    base = written.rsplit("/", 1)[-1]
    for key in folders:
        for listed in sorted(folder_paths.get_filename_list(key)):
            if base and listed.replace("\\", "/").rsplit("/", 1)[-1] == base:
                path = folder_paths.get_full_path(key, listed)
                if path is not None:
                    return path
    raise FileNotFoundError(name)


def pop_extras(state_dict: dict[str, torch.Tensor]):
    """The step-endpoint gate, the stored schedule and the endpoint embedder, which the stock model does not have."""
    gate = state_dict.pop("hyperflow_gate", None)
    sigmas = state_dict.pop("hyperflow_sigmas", None)
    endpoint_sd = {
        name.removeprefix(ENDPOINT_PREFIX): state_dict.pop(name)
        for name in list(state_dict)
        if name.startswith(ENDPOINT_PREFIX)
    }
    return gate, sigmas, endpoint_sd


def packed_model(patcher, path: str) -> PackedModel:
    if patcher is None:
        raise RuntimeError(f"Could not detect the model type of {path}")
    model = patcher.model.diffusion_model
    if not isinstance(model, PackedModel):
        raise RuntimeError(f"{path} is not a packed video+audio DiT checkpoint")
    return model


@functools.cache
def disk_patcher(base: type) -> type:
    """A GGUFModelPatcher whose offloaded weights stay in the memory-mapped file.

    When the transformer does not fit on the GPU, the stock patcher copies every weight into RAM ("release mmap") and
    pins what it offloads, so 9 GB of memory the OS cannot reclaim. Here the weights that do not fit stay in the file's
    page cache and stream to the GPU layer by layer (ComfyUI's lowvram path); under memory pressure the OS drops those
    pages and reads them again from disk. What fits on the GPU loads as usual.
    """

    class HyperFlowDiskPatcher(base):
        mmap_released = True

        def pin_weight_to_device(self, key):
            pass

        def clone(self, *args, **kwargs):
            clone = base.clone(self, *args, **kwargs)
            clone.__class__ = HyperFlowDiskPatcher
            # GGUFModelPatcher.clone copies the source's flag into the instance, which would hide the class's.
            clone.mmap_released = True
            return clone

    return HyperFlowDiskPatcher


def weights_on_disk(path: str, weights_in: str) -> bool:
    if weights_in != "auto":
        return weights_in == "disk"
    try:
        import comfy.storage
    except ImportError:
        return False
    model_fast_disk = getattr(comfy.storage, "model_fast_disk", None)
    return model_fast_disk is not None and model_fast_disk([path])


def load_gguf(path: str, weights_in: str = "auto"):
    """A HyperFlow GGUF as a ComfyUI model, with its factorized blocks and its second time input in place."""
    state_dict = read(path)
    gate, sigmas, endpoint_sd = pop_extras(state_dict)
    plan = FactorizedPlan(state_dict, gguf_placeholder)
    ops = operations()
    patcher = comfy.sd.load_diffusion_model_state_dict(
        state_dict, model_options={"custom_operations": ops}, disable_dynamic=True
    )
    model = packed_model(patcher, path)
    factorized = plan.install(model, ops)
    if gate is not None:
        install_two_time(model, endpoint_sd, float(to_float(gate).flatten()[0]), ops)
    model.hyperflow_sigmas = step_sigmas() if sigmas is None else to_float(sigmas).flatten()
    logging.info(
        "HyperFlow: %d factorized blocks, two-time=%s, %d-step schedule",
        factorized,
        gate is not None,
        model.hyperflow_sigmas.shape[0] - 1,
    )
    on_disk = weights_on_disk(path, weights_in)
    base = _ggml().ops.GGUFModelPatcher
    patcher_class = disk_patcher(base) if on_disk else base
    patcher = patcher_class.clone(patcher)
    patcher.size = 0
    logging.info("HyperFlow: transformer weights in %s (weights_in=%s)", "disk" if on_disk else "ram", weights_in)
    return patcher


def load_safetensors(path: str):
    """A HyperFlow INT8 safetensors file as a ComfyUI model: ComfyUI loads its dense layers and runs the INT8 ones
    on its own kernels; the factorized blocks and the second time input are put in place here."""
    info = checked_info(path)
    state_dict, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
    gate, sigmas, endpoint_sd = pop_extras(state_dict)
    if gate is None or sigmas is None or not endpoint_sd:
        raise RuntimeError(f"{path} has HyperFlow metadata but no step-endpoint tensors")
    plan = FactorizedPlan(state_dict, dense_placeholder)
    full_precision = float32_tensors(state_dict)
    patcher = comfy.sd.load_diffusion_model_state_dict(
        state_dict, model_options={}, metadata=metadata, disable_dynamic=True
    )
    model = packed_model(patcher, path)
    ops = SimpleNamespace(Linear=type(model.condition_proj))
    factorized = plan.install(model, ops)
    install_two_time(model, endpoint_sd, float(gate.float().flatten()[0]), ops)
    restored = keep_float32(model, full_precision)
    restored += keep_float32(model.endpoint_time_embedder, endpoint_sd)
    chunked = limit_int8_rows(model)
    model.hyperflow_sigmas = sigmas.float().flatten().cpu()
    patcher.size = 0
    logging.info(
        "HyperFlow: %s %s (layout %s), %d factorized blocks, %d fp32 tensors, %d row-chunked INT8 modules, "
        "%d-step schedule",
        info.get("variant", "?"),
        info.get("weights", "?"),
        info.get("format"),
        factorized,
        restored,
        chunked,
        model.hyperflow_sigmas.shape[0] - 1,
    )
    return patcher


def load(path: str, weights_in: str = "auto"):
    """A HyperFlow checkpoint as a ComfyUI model. `weights_in` applies to a .gguf only."""
    if path.endswith(".gguf"):
        return load_gguf(path, weights_in)
    return load_safetensors(path)


def sigmas(model, steps: int = DEFAULT_STEPS) -> torch.Tensor:
    """The checkpoint's own schedule for its default step count, or the standard one for 4, 6 or 8 steps."""
    steps = int(steps)
    if steps not in RAW_SIGMAS:
        raise ValueError(f"HyperFlow samples {sorted(RAW_SIGMAS)} steps, got {steps}")
    schedule = getattr(model.model.diffusion_model, "hyperflow_sigmas", None)
    if steps != DEFAULT_STEPS or schedule is None:
        schedule = step_sigmas(steps)
    return schedule.clone().cpu()

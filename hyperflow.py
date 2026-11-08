"""HyperFlow H3 Flash: load its GGUF checkpoints, and the step schedule they sample.

HyperFlow H3 Flash is MiniMax H3 distilled to a few steps. Some of its blocks are pruned, some
linears are factorized into two low-rank halves, and it takes a second time input: the endpoint r
of each step beside its time t. The loader rebuilds that on ComfyUI's own MiniMax H3 model.

- Factorized linears `<name>.down` / `<name>.up` replace the dense `attn.qkv_proj`, `attn.out_proj`, `mlp.fc1`
  and `mlp.fc2` of the blocks that store them; split `q/k/v_proj` are concatenated back into one projection.
- Two-time checkpoints embed the step endpoint r next to t: temb = te(t) + gate * (ee(r) - te(t)).
  Text and video use (1 - sigma, 1 - sigma_next); audio uses the same pair mapped to its own shift; condition
  rows keep r = t. sigma_next comes from the running schedule in `transformer_options["sample_sigmas"]`.

Adapted from ComfyUI-HyperFlow-Flash by Video Rebirth
(https://github.com/Video-Rebirth/ComfyUI-HyperFlow-Flash, commit 20a22ab), under the GNU General
Public License v3.0. See NOTICE. Changed for H3 Take on 2026-11-08: its gguf_io.py, model_patch.py
and loader_nodes.py are joined in this module; GGUF tensors and layers come from this pack's
ggml_ops and ggml_dequant, instead of a ComfyUI-GGUF install found at run time; the folder of
.gguf models is registered here; the nodes are H3 HyperFlow Loader and H3 HyperFlow Sigmas in
nodes.py.
"""

from __future__ import annotations

import contextlib
import functools
import logging
import re
import warnings
from collections import defaultdict

import comfy.ldm.minimax.model as packed_dit
import comfy.sd
import folder_paths
import gguf
import torch
from torch import nn

if __package__:
    from . import ggml_dequant, ggml_ops
else:  # imported on its own, as the tests do
    import ggml_dequant
    import ggml_ops

# The .gguf files in the diffusion model folders, listed under the key ComfyUI-GGUF uses too, so
# either pack can register it first and both see the same files.
GGUF_FOLDER = "unet_gguf"


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
# tensors are read here and handed to the on-the-fly dequantizing ops.

TENSOR_PREFIX = "model.diffusion_model."
TORCH_TYPES = {gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16}


def operations():
    ops = ggml_ops.GGMLOps()
    ops.Linear.dequant_dtype = None
    ops.Linear.patch_dtype = None
    return ops


def placeholder(like: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """A zero-byte weight with the dense shape; the module holding it is replaced before any forward."""
    return ggml_ops.GGMLTensor(torch.empty(0, dtype=torch.uint8), tensor_type=like.tensor_type, tensor_shape=torch.Size(shape))


def to_float(tensor: torch.Tensor) -> torch.Tensor:
    return ggml_dequant.dequantize_tensor(tensor, dtype=torch.float32).reshape(tensor.tensor_shape)


def read(path: str) -> dict[str, torch.Tensor]:
    reader = gguf.GGUFReader(path)
    state_dict = {}
    for tensor in reader.tensors:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The given NumPy array is not writable")
            data = torch.from_numpy(tensor.data)
        shape = torch.Size(int(size) for size in reversed(tensor.shape))
        if tensor.tensor_type in TORCH_TYPES:
            data = data.view(*shape)
        value = ggml_ops.GGMLTensor(data, tensor_type=tensor.tensor_type, tensor_shape=shape)
        if len(shape) <= 1 and tensor.tensor_type == gguf.GGMLQuantizationType.BF16:
            value = ggml_dequant.dequantize_tensor(value, dtype=torch.float32)
        state_dict[tensor.name.removeprefix(TENSOR_PREFIX)] = value
    quantized = [name for name, value in state_dict.items() if ggml_dequant.is_quantized(value)]
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


# HyperFlow's additions on top of ComfyUI's packed video+audio DiT.

PackedModel = packed_dit.MiniMaxH3Model
VIDEO_SHIFT_OPTION = "minimax_h3_sigma_shift_video"
AUDIO_SHIFT_OPTION = "minimax_h3_sigma_shift_audio"
ENDPOINT_PREFIX = "endpoint_time_embedder."
PART = re.compile(r"^blocks\.(\d+)\.(attn\.[qkv]_proj|attn\.out_proj|mlp\.fc[12])\.(down\.weight|up\.weight|weight)$")
QKV_PARTS = ("attn.q_proj", "attn.k_proj", "attn.v_proj")
# Rows of the first step share t = 0 across video and audio but not r; audio moves to the next float32 value.
AUDIO_TIME_OFFSET = 1e-6


class LowRankLinear(nn.Module):
    def __init__(self, down: nn.Module, up: nn.Module) -> None:
        super().__init__()
        self.down = down
        self.up = up

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(x))


class SplitQKV(nn.Module):
    def __init__(self, q_proj: nn.Module, k_proj: nn.Module, v_proj: nn.Module) -> None:
        super().__init__()
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat((self.q_proj(x), self.k_proj(x), self.v_proj(x)), dim=-1)


class FactorizedMLP(packed_dit.MLP):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, hidden = self.fc1(x).chunk(2, dim=-1)
        return self.fc2(nn.functional.silu(gate) * hidden)


def _shape(tensor: torch.Tensor) -> tuple[int, int]:
    return tuple(int(size) for size in tensor.tensor_shape)


class FactorizedPlan:
    """Pops the factorized tensors of every block and leaves dense placeholders for model detection."""

    def __init__(self, state_dict: dict[str, torch.Tensor]) -> None:
        parts: dict[int, dict[str, dict[str, torch.Tensor]]] = defaultdict(lambda: defaultdict(dict))
        for name in list(state_dict):
            match = PART.match(name)
            if match is None:
                continue
            block, module, leaf = int(match.group(1)), match.group(2), match.group(3)
            if module in QKV_PARTS or leaf != "weight":
                parts[block][module][leaf.removesuffix(".weight")] = state_dict.pop(name)
        self.parts = {block: dict(modules) for block, modules in parts.items()}
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
                    linear.load_state_dict({"weight": part["weight"]}, strict=False)
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
    for key in (GGUF_FOLDER, "diffusion_models", "unet"):
        if key in folder_paths.folder_names_and_paths:
            names.update(name for name in folder_paths.get_filename_list(key) if name.endswith(".gguf"))
    return sorted(names)


def gguf_path(name: str) -> str:
    for key in (GGUF_FOLDER, "diffusion_models", "unet"):
        if key in folder_paths.folder_names_and_paths:
            path = folder_paths.get_full_path(key, name)
            if path is not None:
                return path
    raise FileNotFoundError(name)


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


def load(path: str, weights_in: str = "auto"):
    """A HyperFlow GGUF as a ComfyUI model, with its factorized blocks and its second time input in place."""
    state_dict = read(path)
    gate = state_dict.pop("hyperflow_gate", None)
    sigmas = state_dict.pop("hyperflow_sigmas", None)
    endpoint_sd = {
        name.removeprefix(ENDPOINT_PREFIX): state_dict.pop(name)
        for name in list(state_dict)
        if name.startswith(ENDPOINT_PREFIX)
    }
    plan = FactorizedPlan(state_dict)
    ops = operations()
    patcher = comfy.sd.load_diffusion_model_state_dict(
        state_dict, model_options={"custom_operations": ops}, disable_dynamic=True
    )
    if patcher is None:
        raise RuntimeError(f"Could not detect the model type of {path}")
    model = patcher.model.diffusion_model
    if not isinstance(model, PackedModel):
        raise RuntimeError(f"{path} is not a packed video+audio DiT checkpoint")
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
    base = ggml_ops.GGUFModelPatcher
    patcher_class = disk_patcher(base) if on_disk else base
    patcher = patcher_class.clone(patcher)
    patcher.size = 0
    logging.info("HyperFlow: transformer weights in %s (weights_in=%s)", "disk" if on_disk else "ram", weights_in)
    return patcher


def sigmas(model, steps: int = DEFAULT_STEPS) -> torch.Tensor:
    """The checkpoint's own schedule for its default step count, or the standard one for 4, 6 or 8 steps."""
    steps = int(steps)
    if steps not in RAW_SIGMAS:
        raise ValueError(f"HyperFlow samples {sorted(RAW_SIGMAS)} steps, got {steps}")
    schedule = getattr(model.model.diffusion_model, "hyperflow_sigmas", None)
    if steps != DEFAULT_STEPS or schedule is None:
        schedule = step_sigmas(steps)
    return schedule.clone().cpu()

"""Apply H3 LoRAs: renumbered for HyperFlow, added at run time on quantized bases, and with
their time-conditioning (adaLN) part re-injected on pruned bases.

Parts of this module are adapted from ComfyUI-MiniMax-H3-Turbo by Larryvrh
(https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo, commit 4274783), licensed under the
Apache License, Version 2.0 (LICENSES/Apache-2.0.txt, NOTICE). Adapted from it: the in-place
bypass LoRA and the adaLN re-injection that adds B @ A @ silu(t_emb) from the bundled
silu(t_emb) grid (data/h3_silu_temb_grid.safetensors, from the same project).

Changed here: several LoRAs on one model share one bypass injection and one adaLN patch per
module, so stacking them is safe; INT8 fused MLP fc2 layers keep ComfyUI's fused SwiGLU INT8
kernel for the base output while adding the LoRA update in float on swiglu(fc1(x)); HyperFlow's
factorized LowRankLinear / SplitQKV layers attach at run time instead of dropping; each LoRA
can restrict itself to a sigma window [end_sigma, start_sigma]; keys with or without a
diffusion_model. prefix both load, and so do keys in the trainers' lora_unet_ form; H3 blocks
are renumbered for HyperFlow H3 Flash, whose GGUF dense weights take the merge path and whose
adaLN is skipped; the silu(t_emb) grid is kept on each device once; the older unique-timestep
path, the sampler and the debug output are left out.
"""

import json
import logging
import os
import re

import torch
import torch.nn.functional as F

import comfy.lora
import comfy.model_management
import comfy.ops
import comfy.patcher_extension
import comfy.utils
import comfy.weight_adapter

try:
    from .policy import H3_BLOCKS, HYPERFLOW_DROPPED, dotted_lora_key, hyperflow_lora_key, premapped_blocks
except ImportError:
    from policy import H3_BLOCKS, HYPERFLOW_DROPPED, dotted_lora_key, hyperflow_lora_key, premapped_blocks

GRID_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "h3_silu_temb_grid.safetensors")
PREFIXES = ("model.diffusion_model.", "diffusion_model.")
LORA_PART = re.compile(r"\.(lora_A|lora_B|lora_down|lora_up|lora_mid|alpha|dora_scale)(\.weight)?$")
FACTORIZED = ("LowRankLinear", "SplitQKV")
ROW_ELEMENTS = 1 << 30
FULL_WINDOW = (1.0, 0.0)
SIGMA_EPS = 1e-5
STACK = "h3_lora_stack"
INJECTION = "h3_lora"
_GRIDS = {}


def silu_grid(device):
    """The silu(t_emb) of a full H3 at 1025 evenly spaced t, kept once per device."""
    key = str(device)
    if key not in _GRIDS:
        if "cpu" not in _GRIDS:
            _GRIDS["cpu"] = comfy.utils.load_torch_file(GRID_FILE)["silu_t_emb_grid"].float()
        _GRIDS[key] = _GRIDS["cpu"].to(device)
    return _GRIDS[key]


def read_lora(path):
    """A LoRA's tensors, with any diffusion_model. prefix taken off its keys."""
    lora = {}
    for key, value in comfy.utils.load_torch_file(path, safe_load=True).items():
        for prefix in PREFIXES:
            if key.startswith(prefix):
                key = key[len(prefix):]
                break
        lora[key] = value
    return lora


def lora_metadata(path):
    """The metadata a safetensors LoRA carries in its header, or nothing when it has none or is another kind of file."""
    try:
        with open(path, "rb") as file:
            size = int.from_bytes(file.read(8), "little")
            if not 0 < size < 100 * 1024 * 1024:
                return {}
            return json.loads(file.read(size)).get("__metadata__") or {}
    except (OSError, ValueError, AttributeError):
        return {}


def module_names(lora):
    """The layers a LoRA changes, such as blocks.3.attn.qkv_proj."""
    names = set()
    for key in lora:
        match = LORA_PART.search(key)
        if match:
            names.add(key[:match.start()])
    return sorted(names)


def is_gguf(dm):
    """True when the model's weights are GGUF tensors, as HyperFlow's are."""
    for index, parameter in enumerate(dm.parameters()):
        if type(parameter).__name__ == "GGMLTensor":
            return True
        if index > 64:
            return False
    return False


def factorized_layers(dm):
    """Layers HyperFlow stores in low-rank or split parts instead of one .weight tensor."""
    return {name for name, module in dm.named_modules() if type(module).__name__ in FACTORIZED}


def is_int8_fused_fc2(dm, module_path):
    """True when module_path is a stock MLP.fc2 whose INT8 weight runs through linear_input_act."""
    if not module_path.endswith(".mlp.fc2"):
        return False
    try:
        mlp = comfy.utils.get_attr(dm, module_path[:-len(".fc2")])
        weight = comfy.utils.get_attr(dm, module_path + ".weight")
    except AttributeError:
        return False
    if type(mlp).__name__ == "FactorizedMLP":
        return False
    return (getattr(weight, "_layout_cls", None) == "TensorWiseINT8Layout"
            and not getattr(getattr(weight, "_params", None), "transposed", False))


def normalize_window(start_sigma=1.0, end_sigma=0.0):
    """A (high, low) sigma window clamped to [0, 1]."""
    a = 1.0 if start_sigma is None else max(0.0, min(1.0, float(start_sigma)))
    b = 0.0 if end_sigma is None else max(0.0, min(1.0, float(end_sigma)))
    return (max(a, b), min(a, b))


def sigma_active(sigma, window):
    """True when sigma falls inside window=(high, low), or when no step sigma is tracked."""
    if sigma is None or window == FULL_WINDOW:
        return True
    high, low = window
    return (low - SIGMA_EPS) <= sigma <= (high + SIGMA_EPS)


def _swiglu(x):
    gate, hidden = x.chunk(2, dim=-1)
    return F.silu(gate) * hidden


class StackedLoRA(comfy.weight_adapter.LoRAAdapter):
    """One or more plain LoRAs on one Linear or factorized layer, added to its output at run time.

    The updates go straight into the layer's fresh output, so each adds one temporary,
    not the three of ComfyUI's stock bypass. Each entry may carry its own (start, end) sigma window.
    """

    def __init__(self, key, parts, state=None):
        unpacked = [(p[0], p[1], p[2], p[3] if len(p) > 3 else FULL_WINDOW) for p in parts]
        super().__init__({key}, tuple(tensor for up, down, _, _ in unpacked for tensor in (up, down)))
        self.scales = [scale for _, _, scale, _ in unpacked]
        self.windows = [window for _, _, _, window in unpacked]
        self.state = state

    def any_active(self, sigma):
        return any(scale != 0 and sigma_active(sigma, window)
                   for scale, window in zip(self.scales, self.windows))

    def add_to(self, out, x, sigma):
        for index, (scale, window) in enumerate(zip(self.scales, self.windows)):
            if scale == 0 or not sigma_active(sigma, window):
                continue
            up, down = self.weights[2 * index], self.weights[2 * index + 1]
            dx = F.linear(
                F.linear(x, down.to(device=x.device, dtype=x.dtype)),
                up.to(device=x.device, dtype=x.dtype),
            )
            out.add_(dx, alpha=scale)
        return out

    def bypass_forward(self, org_forward, x, *args, **kwargs):
        out = org_forward(x, *args, **kwargs)
        sigma = None if self.state is None else self.state.get("sigma")
        return self.add_to(out, x, sigma)


class MlpFc2Hook:
    """Hook on an MLP whose INT8 fc2 runs through ComfyUI's fused SwiGLU kernel.

    Keeping linear_input_act for the base fc2 call preserves the exact fused INT8 output
    (avoiding unfused re-quantization drift that would swamp a small LoRA), while adding
    the LoRA delta on swiglu(fc1(x)) in float precision.
    """

    def __init__(self, mlp, adapter):
        self.mlp = mlp
        self.adapter = adapter
        self.original_forward = None

    def _inner(self, x):
        sigma = None if self.adapter.state is None else self.adapter.state.get("sigma")
        if not self.adapter.any_active(sigma):
            return comfy.ops.linear_input_act(self.mlp.fc2, self.mlp.fc1(x), "swiglu")
        h = self.mlp.fc1(x)
        out = comfy.ops.linear_input_act(self.mlp.fc2, h, "swiglu")
        return self.adapter.add_to(out, _swiglu(h), sigma)

    def inject(self):
        if self.original_forward is not None:
            return
        device = comfy.model_management.get_torch_device()
        self.adapter.weights = tuple(w.to(device=device) for w in self.adapter.weights)
        self.original_forward = self.mlp.forward
        if hasattr(self.original_forward, "__wrapped__"):
            fc2 = self.mlp.fc2
            rows = max(1, ROW_ELEMENTS // max(fc2.in_features, fc2.out_features))
            inner = self._inner

            def chunked(x, *args, **kwargs):
                flat = x.reshape(-1, x.shape[-1])
                if flat.shape[0] <= rows:
                    return inner(x)
                out = None
                for start in range(0, flat.shape[0], rows):
                    part = inner(flat[start : start + rows])
                    if out is None:
                        out = part.new_empty((flat.shape[0], part.shape[-1]))
                    out[start : start + part.shape[0]] = part
                return out.reshape(*x.shape[:-1], out.shape[-1])

            self.mlp.forward = chunked
        else:
            self.mlp.forward = self._inner

    def eject(self):
        if self.original_forward is None:
            return
        self.mlp.forward = self.original_forward
        self.original_forward = None


def adaln_forward(base, pairs, table, state=None):
    """A forward for one pruned adaLN projection that also adds each LoRA's B @ A @ silu(t_emb).

    The pruned base feeds the projection a point on its small time curve, so the LoRA's
    update, which lives in the full silu(t_emb) space, is added from the matching row of the
    bundled grid. Patched as the forward attribute only, so the module tree stays as it is.
    """
    entries = [(p[0], p[1], p[2] if len(p) > 2 else FULL_WINDOW) for p in pairs]

    def forward(t_emb):
        x = base.linear(F.silu(t_emb) if base.apply_silu else t_emb)
        sigma = None if state is None else state.get("sigma")
        active = [(a, b) for a, b, window in entries if sigma_active(sigma, window)]
        if active:
            rows = torch.cdist(t_emb.detach().float(), table.to(t_emb.device, torch.float32)).argmin(dim=1)
            silu_t = silu_grid(t_emb.device)[rows].to(x.dtype)
            for a, b in active:
                x = x + (b.to(x.device, x.dtype) @ (a.to(x.device, x.dtype) @ silu_t.T)).T
        x = x.view(x.shape[0] * base.modalities, base.expand * base.hidden)
        return x.chunk(base.expand, dim=-1)

    return forward


def apply(model, lora_path, lora_name, strength, mode="auto", start_sigma=1.0, end_sigma=0.0):
    """The model with this LoRA added on top of any H3 LoRAs already on it."""
    if strength == 0:
        return model
    window = normalize_window(start_sigma, end_sigma)
    lora = read_lora(lora_path)
    new = model.clone()
    dm = new.model.diffusion_model
    count = len(dm.blocks) if hasattr(dm, "blocks") else H3_BLOCKS
    hyperflow = count in HYPERFLOW_DROPPED
    notes = []
    # A file published for one HyperFlow model already counts that model's blocks.
    mapped = premapped_blocks(lora_metadata(lora_path))
    if mapped is not None and (not hyperflow or mapped != tuple(sorted(HYPERFLOW_DROPPED[count]))):
        notes.append("this file is numbered for a HyperFlow model that dropped blocks %s, not for this model: "
                     "use the base H3 file here" % ", ".join(str(b) for b in mapped))
    if hyperflow and mapped is None:
        moved = {}
        for key, value in lora.items():
            key = hyperflow_lora_key(key, HYPERFLOW_DROPPED[count])
            if key is not None:
                moved[key] = value
        lora = moved
    elif hyperflow:
        if mapped == tuple(sorted(HYPERFLOW_DROPPED[count])):
            notes.append("already numbered for this HyperFlow model")
    elif count != H3_BLOCKS:
        notes.append("no block map for %s blocks, so the keys load as they are" % count)

    present = set(new.model.state_dict().keys())
    dense = {key[len("diffusion_model."):-len(".weight")] for key in present
             if key.startswith("diffusion_model.") and key.endswith(".weight")}
    factorized = factorized_layers(dm)
    known = dense | factorized
    # A LoRA in the trainer's lora_unet_ form takes the model's own layer names.
    layers = {name.replace(".", "_"): name for name in known}
    lora = {dotted_lora_key(key, layers): value for key, value in lora.items()}

    modules = module_names(lora)
    adaln = [m for m in modules if m.endswith("adaln_proj.linear")]
    pruned = bool(getattr(dm, "use_adaln_curves", False)) and not hyperflow
    backbone = [m for m in modules if m not in adaln or not (pruned or hyperflow)]
    if hyperflow and adaln:
        notes.append("%s adaLN layers skipped: HyperFlow's adaLN takes its own two-time input" % len(adaln))

    missing = sorted(m for m in backbone if m not in known)
    targets = {m: "diffusion_model.%s.weight" % m for m in backbone if m in known}
    # Factorized modules have no single .weight to patch, and sigma-gated LoRAs turn on and off per step.
    merge_dense = (mode == "merge" or is_gguf(dm)) and window == FULL_WINDOW
    merged = {m: key for m, key in targets.items() if merge_dense and m in dense}
    bypass = {}
    for key, adapter in comfy.lora.load_lora(lora, {m: k for m, k in targets.items() if m not in merged},
                                             log_missing=False).items():
        module = key[len("diffusion_model."):-len(".weight")]
        weights = adapter.weights if isinstance(adapter, comfy.weight_adapter.LoRAAdapter) else None
        if weights is None or any(part is not None for part in weights[3:5]):
            # Not a plain LoRA (a LoCon mid or DoRA scale): merge it when the model has a dense weight.
            if module in dense:
                merged[module] = key
            else:
                missing.append(module)
            continue
        up, down, alpha = weights[0], weights[1], weights[2]
        scale = (float(alpha) / down.shape[0] if alpha is not None else 1.0) * strength
        bypass[key] = (up, down, scale, window)
    if merged:
        new.add_patches(comfy.lora.load_lora(lora, merged, log_missing=False), strength)

    pairs = {}
    if pruned:
        for module in adaln:
            down = lora.get(module + ".lora_A.weight", lora.get(module + ".lora_down.weight"))
            up = lora.get(module + ".lora_B.weight", lora.get(module + ".lora_up.weight"))
            if down is None or up is None:
                notes.append("%s is not a plain LoRA, so its adaLN part is skipped" % module)
                continue
            alpha = lora.get(module + ".alpha")
            scale = (float(alpha) / down.shape[0] if alpha is not None else 1.0) * strength
            pairs[module] = (down, up * scale, window)

    stack = list(new.get_attachment(STACK) or [])
    stack.append({"name": lora_name, "bypass": bypass, "adaln": pairs, "window": window})
    new.set_attachments(STACK, stack)
    state = {"sigma": None}
    hooks = rebuild_bypass(new, dm, stack, state)
    rebuild_adaln(new, dm, stack, notes, state)
    rebuild_step_gate(new, stack, hooks, state)
    if window != FULL_WINDOW:
        notes.append("active on sigma %.3f..%.3f" % (window[0], window[1]))

    if missing:
        missing = sorted(set(missing))
        notes.append("%s %s this model does not have: %s" % (
            len(missing), "layer" if len(missing) == 1 else "layers",
            ", ".join(missing[:4]) + (" ..." if len(missing) > 4 else "")))
    logging.info("H3 LoRA %s at %s: %s layers, %s at run time, %s merged, %s adaLN at run time%s",
                 lora_name, strength, len(modules), len(bypass), len(merged), len(pairs),
                 "; " + "; ".join(notes) if notes else "")
    return new


def rebuild_bypass(new, dm, stack, state=None):
    """One runtime hook per layer across all stacked LoRAs, including fused INT8 fc2 and factorized layers."""
    combined = {}
    for entry in stack:
        for key, parts in entry["bypass"].items():
            combined.setdefault(key, []).append(parts)
    hooks = []
    for key, parts in combined.items():
        module_path = key[:-len(".weight")] if key.endswith(".weight") else key
        rel = module_path[len("diffusion_model."):] if module_path.startswith("diffusion_model.") else module_path
        adapter = StackedLoRA(key, parts, state=state)
        if is_int8_fused_fc2(dm, rel):
            mlp_path = module_path[:-len(".fc2")]
            try:
                mlp = new.get_model_object(mlp_path)
            except AttributeError:
                logging.warning("H3 LoRA: MLP not found for %s", mlp_path)
                continue
            hooks.append((MlpFc2Hook(mlp, adapter), adapter))
            continue
        try:
            module = new.get_model_object(module_path)
        except AttributeError:
            logging.warning("H3 LoRA: module not found for %s", module_path)
            continue
        hooks.append((comfy.weight_adapter.BypassForwardHook(module, adapter, multiplier=1.0), adapter))
    if hooks:
        def inject_all(_patcher):
            for hook, _ in hooks:
                hook.inject()

        def eject_all(_patcher):
            for hook, _ in hooks:
                hook.eject()

        new.set_injections(INJECTION, [comfy.patcher_extension.PatcherInjection(inject=inject_all, eject=eject_all)])
    else:
        new.remove_injections(INJECTION)
    return hooks


def rebuild_step_gate(new, stack, hooks, state):
    """Track each step's sigma when any stacked LoRA restricts its sigma window."""
    wrapper_type = comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL
    new.remove_wrappers_with_key(wrapper_type, INJECTION)
    if not any(entry.get("window", FULL_WINDOW) != FULL_WINDOW for entry in stack):
        return

    def step_wrapper(executor, x, timestep, *args, **kwargs):
        sigma = float(timestep.flatten()[0]) / 1000.0
        state["sigma"] = sigma
        for hook, adapter in hooks:
            if hasattr(hook, "module") and hook.original_forward is not None:
                hook.module.comfy_force_forward = adapter.any_active(sigma) or hook.original_force_forward
        try:
            return executor(x, timestep, *args, **kwargs)
        finally:
            state["sigma"] = None
            for hook, _ in hooks:
                if hasattr(hook, "module") and hook.original_forward is not None:
                    hook.module.comfy_force_forward = True

    new.add_wrapper_with_key(wrapper_type, INJECTION, step_wrapper)


def rebuild_adaln(new, dm, stack, notes, state=None):
    """One adaLN patch per projection for every LoRA on a pruned base."""
    combined = {}
    for entry in stack:
        for module, pair in entry["adaln"].items():
            combined.setdefault(module, []).append(pair)
    if not combined:
        return 0
    table = getattr(dm, "adaln_t_table", None)
    if table is None or table.shape[0] != silu_grid("cpu").shape[0]:
        notes.append("adaLN skipped: this base's time curve does not match the bundled grid")
        return 0
    for module, pairs in combined.items():
        key = "diffusion_model." + module.rsplit(".linear", 1)[0]
        new.add_object_patch(key + ".forward", adaln_forward(new.get_model_object(key), pairs, table, state=state))
    return len(combined)

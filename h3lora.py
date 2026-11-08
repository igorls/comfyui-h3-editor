"""Apply H3 LoRAs: renumbered for HyperFlow, added at run time on quantized bases, and with
their time-conditioning (adaLN) part re-injected on pruned bases.

Parts of this module are adapted from ComfyUI-MiniMax-H3-Turbo by Larryvrh
(https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo, commit 4274783), licensed under the
Apache License, Version 2.0 (LICENSES/Apache-2.0.txt, NOTICE). Adapted from it: the in-place
bypass LoRA, the rule that int8 fused fc2 weights take the merge path, and the adaLN
re-injection that adds B @ A @ silu(t_emb) from the bundled silu(t_emb) grid
(data/h3_silu_temb_grid.safetensors, from the same project).

Changed here: several LoRAs on one model share one bypass injection and one adaLN patch per
module, so stacking them is safe; keys with or without a diffusion_model. prefix both load; H3
blocks are renumbered for HyperFlow H3 Flash, whose weights take the merge path and whose
adaLN is skipped; the silu(t_emb) grid is kept on each device once; the older unique-timestep
path, the sampler and the debug output are left out.
"""

import logging
import os
import re

import torch
import torch.nn.functional as F

import comfy.lora
import comfy.utils
import comfy.weight_adapter

try:
    from .policy import H3_BLOCKS, HYPERFLOW_DROPPED, hyperflow_lora_key
except ImportError:
    from policy import H3_BLOCKS, HYPERFLOW_DROPPED, hyperflow_lora_key

GRID_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "h3_silu_temb_grid.safetensors")
PREFIXES = ("model.diffusion_model.", "diffusion_model.")
LORA_PART = re.compile(r"\.(lora_A|lora_B|lora_down|lora_up|lora_mid|alpha|dora_scale)(\.weight)?$")
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


def int8_fused_fc2(dm, modules):
    """MLP fc2 layers whose int8 weight ComfyUI multiplies in a fused kernel.

    That kernel reads the weight directly and never calls the layer's forward, so a bypass
    hook on it would never run. Those layers take the merge path instead.
    """
    fused = []
    for module in modules:
        if not module.endswith(".mlp.fc2"):
            continue
        try:
            weight = comfy.utils.get_attr(dm, module + ".weight")
        except AttributeError:
            continue
        if (getattr(weight, "_layout_cls", None) == "TensorWiseINT8Layout"
                and not getattr(getattr(weight, "_params", None), "transposed", False)):
            fused.append(module)
    return fused


class StackedLoRA(comfy.weight_adapter.LoRAAdapter):
    """One or more plain LoRAs on one Linear layer, added to its output at run time.

    The updates go straight into the layer's fresh output, so each adds one temporary,
    not the three of ComfyUI's stock bypass.
    """

    def __init__(self, key, parts):
        super().__init__({key}, tuple(tensor for up, down, _ in parts for tensor in (up, down)))
        self.scales = [scale for _, _, scale in parts]

    def bypass_forward(self, org_forward, x, *args, **kwargs):
        out = org_forward(x, *args, **kwargs)
        for index, scale in enumerate(self.scales):
            up, down = self.weights[2 * index], self.weights[2 * index + 1]
            out.add_(F.linear(F.linear(x, down.to(dtype=x.dtype)), up.to(dtype=x.dtype)), alpha=scale)
        return out


def adaln_forward(base, pairs, table):
    """A forward for one pruned adaLN projection that also adds each LoRA's B @ A @ silu(t_emb).

    The pruned base feeds the projection a point on its small time curve, so the LoRA's
    update, which lives in the full silu(t_emb) space, is added from the matching row of the
    bundled grid. Patched as the forward attribute only, so the module tree stays as it is.
    """

    def forward(t_emb):
        x = base.linear(F.silu(t_emb) if base.apply_silu else t_emb)
        rows = torch.cdist(t_emb.detach().float(), table.to(t_emb.device, torch.float32)).argmin(dim=1)
        silu_t = silu_grid(t_emb.device)[rows].to(x.dtype)
        for a, b in pairs:
            x = x + (b.to(x.device, x.dtype) @ (a.to(x.device, x.dtype) @ silu_t.T)).T
        x = x.view(x.shape[0] * base.modalities, base.expand * base.hidden)
        return x.chunk(base.expand, dim=-1)

    return forward


def apply(model, lora_path, lora_name, strength, mode="auto"):
    """The model with this LoRA added on top of any H3 LoRAs already on it."""
    if strength == 0:
        return model
    lora = read_lora(lora_path)
    new = model.clone()
    dm = new.model.diffusion_model
    count = len(dm.blocks) if hasattr(dm, "blocks") else H3_BLOCKS
    hyperflow = count in HYPERFLOW_DROPPED
    notes = []
    if hyperflow:
        moved = {}
        for key, value in lora.items():
            key = hyperflow_lora_key(key, HYPERFLOW_DROPPED[count])
            if key is not None:
                moved[key] = value
        lora = moved
    elif count != H3_BLOCKS:
        notes.append("no block map for %s blocks, so the keys load as they are" % count)

    modules = module_names(lora)
    adaln = [m for m in modules if m.endswith("adaln_proj.linear")]
    pruned = bool(getattr(dm, "use_adaln_curves", False)) and not hyperflow
    backbone = [m for m in modules if m not in adaln or not (pruned or hyperflow)]
    if hyperflow and adaln:
        notes.append("%s adaLN layers skipped: HyperFlow's adaLN takes its own two-time input" % len(adaln))

    present = set(new.model.state_dict().keys())
    targets = {m: "diffusion_model.%s.weight" % m for m in backbone}
    missing = sorted(m for m, key in targets.items() if key not in present)
    targets = {m: key for m, key in targets.items() if key in present}
    merge_all = mode == "merge" or is_gguf(dm)
    fused = set() if merge_all else set(int8_fused_fc2(dm, targets))
    merged = {m: key for m, key in targets.items() if merge_all or m in fused}
    bypass = {}
    for key, adapter in comfy.lora.load_lora(lora, {m: k for m, k in targets.items() if m not in merged},
                                             log_missing=False).items():
        weights = adapter.weights if isinstance(adapter, comfy.weight_adapter.LoRAAdapter) else None
        if weights is None or any(part is not None for part in weights[3:5]):
            # Not a plain LoRA (a LoCon mid or DoRA scale): merge it instead.
            module = key[len("diffusion_model."):-len(".weight")]
            merged[module] = key
            continue
        up, down, alpha = weights[0], weights[1], weights[2]
        scale = (float(alpha) / down.shape[0] if alpha is not None else 1.0) * strength
        bypass[key] = (up, down, scale)
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
            pairs[module] = (down, up * scale)

    stack = list(new.get_attachment(STACK) or [])
    stack.append({"name": lora_name, "bypass": bypass, "adaln": pairs})
    new.set_attachments(STACK, stack)
    rebuild_bypass(new, stack)
    rebuild_adaln(new, dm, stack, notes)

    if missing:
        notes.append("%s %s this model does not have whole: %s" % (
            len(missing), "layer" if len(missing) == 1 else "layers", ", ".join(missing[:4]) + (" ..." if len(missing) > 4 else "")))
    logging.info("H3 LoRA %s at %s: %s layers, %s at run time, %s merged, %s adaLN at run time%s",
                 lora_name, strength, len(modules), len(bypass), len(merged), len(pairs),
                 "; " + "; ".join(notes) if notes else "")
    return new


def rebuild_bypass(new, stack):
    """One bypass injection for every LoRA on the model, so layers several touch get one hook."""
    combined = {}
    for entry in stack:
        for key, parts in entry["bypass"].items():
            combined.setdefault(key, []).append(parts)
    manager = comfy.weight_adapter.BypassInjectionManager()
    for key, parts in combined.items():
        manager.add_adapter(key, StackedLoRA(key, parts), strength=1.0)
    injections = manager.create_injections(new.model)
    if manager.get_hook_count() > 0:
        new.set_injections(INJECTION, injections)
    else:
        new.remove_injections(INJECTION)
    return manager.get_hook_count()


def rebuild_adaln(new, dm, stack, notes):
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
        new.add_object_patch(key + ".forward", adaln_forward(new.get_model_object(key), pairs, table))
    return len(combined)

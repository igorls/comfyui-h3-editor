"""Hybrid block-swapping loader for MiniMax H3 safetensors checkpoints.

Loads a primary diffusion model (typically fl2va) and selectively overlays matching
transformer block tensor families from a secondary checkpoint (typically ref2va)
before handing the merged state dict to ComfyUI's diffusion model loader.

Pure-Python header inspection, block parsing, and tensor-family validation live here
without top-level ComfyUI or PyTorch imports so they can be tested and validated in
the editor without loading any weights.
"""

from __future__ import annotations

import json
import logging
import os
import re
import struct
from dataclasses import dataclass

try:
    from .policy import H3_BLOCKS
except ImportError:
    from policy import H3_BLOCKS

SCOPES = ("adaln", "all", "attn", "mlp", "attn_mlp")
SCOPE_LABELS = {
    "adaln": "Adaptive LayerNorm (adaln)",
    "all": "All block tensors (all)",
    "attn": "Attention (attn)",
    "mlp": "MLP / FFN (mlp)",
    "attn_mlp": "Attention + MLP (attn_mlp)",
}
WEIGHT_DTYPES = ("default", "fp8_e4m3fn", "fp8_e4m3fn_fast", "fp8_e5m2")
DEFAULT_BLOCKS = "45-49"
DEFAULT_SCOPE = "adaln"
MAX_HEADER_BYTES = 128 << 20

PREFIXES = ("model.diffusion_model.", "diffusion_model.")
BLOCK_RE = re.compile(r"^(?:.*?\.)?blocks\.(\d+)\.(.+)$")

# Suffixes that belong to the same logical parameter family (weights, biases, and quantization siblings).
FAMILY_SUFFIXES = (
    ".weight_scale_2",
    ".weight_scale",
    ".input_scale",
    ".scale_weight",
    ".scale_input",
    ".comfy_quant",
    ".zero_point",
    ".qscale",
    ".weight",
    ".bias",
)

ADALN_RE = re.compile(r"(?:^|[._])(adaln|ada_ln|modulation|scale_shift|time_embed|temb|norm\d*_mod)(?:$|[._])", re.IGNORECASE)
ATTN_RE = re.compile(r"(?:^|[._])(attn|self_attn|cross_attn|qkv|qkv_proj|q_proj|k_proj|v_proj|out_proj|o_proj|q_norm|k_norm)(?:$|[._])", re.IGNORECASE)
MLP_RE = re.compile(r"(?:^|[._])(mlp|ffn|fc1|fc2|gate_proj|up_proj|down_proj|w1|w2|w3)(?:$|[._])", re.IGNORECASE)


@dataclass(frozen=True)
class HybridPlan:
    """Validated swap plan built from safetensors headers alone."""

    swaps: tuple[tuple[str, str], ...]
    blocks_matched: tuple[int, ...]
    families_swapped: int
    skipped: tuple[str, ...]


def parse_blocks(spec: str, total: int = H3_BLOCKS) -> tuple[int, ...]:
    """Parse a block specification such as '45-49' or '0-10, 45-49' or 'all' into sorted block indices."""
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("blocks must be a non-empty range such as '45-49' or '0-10, 45-49'")
    raw = spec.strip().lower()
    if raw == "all":
        return tuple(range(total))
    indices: set[int] = set()
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            pieces = token.split("-")
            if len(pieces) != 2 or not pieces[0].strip().isdigit() or not pieces[1].strip().isdigit():
                raise ValueError("invalid block range %r (use e.g. '45-49' or '0-10, 45-49')" % token)
            lo, hi = int(pieces[0].strip()), int(pieces[1].strip())
            if lo > hi:
                raise ValueError("reversed block range %r (start %d > end %d)" % (token, lo, hi))
            if lo < 0 or hi >= total:
                raise ValueError("block range %r is out of bounds (0..%d)" % (token, total - 1))
            indices.update(range(lo, hi + 1))
        else:
            if not token.isdigit():
                raise ValueError("invalid block index %r (use integers 0..%d)" % (token, total - 1))
            idx = int(token)
            if idx < 0 or idx >= total:
                raise ValueError("block index %d is out of bounds (0..%d)" % (idx, total - 1))
            indices.add(idx)
    if not indices:
        raise ValueError("blocks must name at least one block index")
    return tuple(sorted(indices))


def format_blocks(indices: tuple[int, ...] | list[int]) -> str:
    """Format sorted block indices back into compact ranges like '0-10, 45-49'."""
    nums = sorted(set(int(i) for i in indices))
    if not nums:
        return ""
    ranges = []
    start = end = nums[0]
    for n in nums[1:]:
        if n == end + 1:
            end = n
        else:
            ranges.append(f"{start}-{end}" if start != end else str(start))
            start = end = n
    ranges.append(f"{start}-{end}" if start != end else str(start))
    return ", ".join(ranges)


def strip_prefix(key: str) -> str:
    for prefix in PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def block_index(canonical_key: str) -> tuple[int, str] | None:
    """Return (block_index, remainder_after_block) or None if the key is not inside a transformer block."""
    match = BLOCK_RE.match(canonical_key)
    if not match:
        return None
    return int(match.group(1)), match.group(2)


def block_count(header: dict[str, dict]) -> int:
    """Number of transformer blocks present in a safetensors header (max block index + 1), or H3_BLOCKS if none."""
    highest = -1
    for key in header:
        parsed = block_index(strip_prefix(key))
        if parsed is not None and parsed[0] > highest:
            highest = parsed[0]
    return highest + 1 if highest >= 0 else H3_BLOCKS


def matches_scope(remainder: str, scope: str) -> bool:
    """Whether a tensor path inside a block (after 'blocks.<n>.') belongs to `scope`."""
    if scope == "all":
        return True
    if scope == "adaln":
        return bool(ADALN_RE.search(remainder))
    if scope == "attn":
        return bool(ATTN_RE.search(remainder))
    if scope == "mlp":
        return bool(MLP_RE.search(remainder))
    if scope == "attn_mlp":
        return bool(ATTN_RE.search(remainder) or MLP_RE.search(remainder))
    raise ValueError("scope must be one of %s, not %r" % (", ".join(SCOPES), scope))


def tensor_family(canonical_key: str) -> str:
    """Logical parameter family stem so weight, bias, scale, and .comfy_quant always move together."""
    for suffix in FAMILY_SUFFIXES:
        if canonical_key.endswith(suffix):
            return canonical_key[: -len(suffix)]
    return canonical_key


def read_safetensors_header(path: str) -> dict[str, dict]:
    """Read tensor metadata entries from a safetensors header without loading any weights."""
    if not path.lower().endswith(".safetensors"):
        raise ValueError("%s is not a .safetensors file; hybrid block swapping requires safetensors checkpoints" % os.path.basename(path))
    with open(path, "rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise ValueError("%s is not a valid safetensors file" % os.path.basename(path))
        header_len = struct.unpack("<Q", raw_len)[0]
        if not 0 < header_len <= MAX_HEADER_BYTES:
            raise ValueError("%s has an invalid safetensors header size (%d bytes)" % (os.path.basename(path), header_len))
        header = json.loads(handle.read(header_len).decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("%s has a malformed safetensors header" % os.path.basename(path))
    return {k: v for k, v in header.items() if k != "__metadata__" and isinstance(v, dict)}


def _group_families(header: dict[str, dict]) -> dict[str, dict[str, tuple[str, dict]]]:
    """Group header entries by tensor family -> {canonical_key: (original_key, meta)}."""
    families: dict[str, dict[str, tuple[str, dict]]] = {}
    for orig_key, meta in header.items():
        canon = strip_prefix(orig_key)
        fam = tensor_family(canon)
        families.setdefault(fam, {})[canon] = (orig_key, meta)
    return families


def plan_swap(base_header: dict[str, dict], overlay_header: dict[str, dict], blocks: str, scope: str = DEFAULT_SCOPE) -> HybridPlan:
    """Build a deterministic, quantization-safe swap plan from two safetensors headers."""
    if scope not in SCOPES:
        raise ValueError("scope must be one of %s, not %r" % (", ".join(SCOPES), scope))
    selected = set(parse_blocks(blocks))
    base_fams = _group_families(base_header)
    overlay_fams = _group_families(overlay_header)

    swaps: list[tuple[str, str]] = []
    matched_blocks: set[int] = set()
    skipped: list[str] = []
    families_swapped = 0

    for fam in sorted(base_fams):
        # Use any canonical key in the family to check block index and scope.
        sample_canon = next(iter(base_fams[fam]))
        parsed = block_index(sample_canon)
        if parsed is None:
            continue
        idx, remainder = parsed
        if idx not in selected or not matches_scope(remainder, scope):
            continue

        b_members = base_fams[fam]
        o_members = overlay_fams.get(fam)
        if not o_members:
            skipped.append(f"{fam}: missing in overlay")
            continue
        if set(b_members.keys()) != set(o_members.keys()):
            skipped.append(f"{fam}: quantization/sibling keys differ")
            continue

        compatible = True
        fam_pairs: list[tuple[str, str]] = []
        for canon in sorted(b_members):
            b_orig, b_meta = b_members[canon]
            o_orig, o_meta = o_members[canon]
            if b_meta.get("shape") != o_meta.get("shape") or b_meta.get("dtype") != o_meta.get("dtype"):
                skipped.append(
                    f"{canon}: shape/dtype mismatch ({b_meta.get('shape')}/{b_meta.get('dtype')} vs "
                    f"{o_meta.get('shape')}/{o_meta.get('dtype')})"
                )
                compatible = False
                break
            fam_pairs.append((b_orig, o_orig))

        if compatible:
            swaps.extend(fam_pairs)
            matched_blocks.add(idx)
            families_swapped += 1

    if not swaps:
        detail = (" (" + "; ".join(skipped[:3]) + ")" if skipped else "")
        raise ValueError(
            f"No compatible tensors matched blocks={blocks!r}, scope={scope!r}{detail}. "
            "Check that both checkpoints use the same quantization format and block layout."
        )

    return HybridPlan(
        swaps=tuple(swaps),
        blocks_matched=tuple(sorted(matched_blocks)),
        families_swapped=families_swapped,
        skipped=tuple(skipped),
    )


def resolve_model_path(name: str) -> str:
    """Resolve a diffusion model filename from ComfyUI's diffusion_models / unet folders."""
    import folder_paths

    for folder in ("diffusion_models", "unet"):
        if folder in folder_paths.folder_names_and_paths:
            path = folder_paths.get_full_path(folder, name)
            if path and os.path.isfile(path):
                return path
    raise FileNotFoundError(f"{name} was not found in ComfyUI's diffusion_models folder")


def load(base_name: str, overlay_name: str, blocks: str = DEFAULT_BLOCKS, scope: str = DEFAULT_SCOPE,
         weight_dtype: str = "default"):
    """Load `base_name` and selectively overlay matching block tensor families from `overlay_name`."""
    import comfy.sd
    import comfy.utils
    import safetensors
    import torch

    if weight_dtype not in WEIGHT_DTYPES:
        raise ValueError("weight_dtype must be one of %s, not %r" % (", ".join(WEIGHT_DTYPES), weight_dtype))

    base_path = resolve_model_path(base_name)
    overlay_path = resolve_model_path(overlay_name)

    base_header = read_safetensors_header(base_path)
    overlay_header = read_safetensors_header(overlay_path)
    plan = plan_swap(base_header, overlay_header, blocks, scope)

    state_dict, metadata = comfy.utils.load_torch_file(base_path, return_metadata=True)
    with safetensors.safe_open(overlay_path, framework="pt", device="cpu") as overlay_file:
        for base_key, overlay_key in plan.swaps:
            state_dict[base_key] = overlay_file.get_tensor(overlay_key)

    model_options = {}
    if weight_dtype == "fp8_e4m3fn":
        model_options["dtype"] = torch.float8_e4m3fn
    elif weight_dtype == "fp8_e4m3fn_fast":
        model_options["dtype"] = torch.float8_e4m3fn
        model_options["fp8_optimizations"] = True
    elif weight_dtype == "fp8_e5m2":
        model_options["dtype"] = torch.float8_e5m2

    logging.info(
        "[H3HybridLoader] %s + %s: swapped %d tensors (%d families) across %d blocks (%s, scope=%s)%s",
        os.path.basename(base_path),
        os.path.basename(overlay_path),
        len(plan.swaps),
        plan.families_swapped,
        len(plan.blocks_matched),
        format_blocks(plan.blocks_matched),
        scope,
        f" [{len(plan.skipped)} skipped]" if plan.skipped else "",
    )
    return comfy.sd.load_diffusion_model_state_dict(state_dict, model_options=model_options, metadata=metadata)

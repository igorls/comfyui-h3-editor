"""Calibrate a Universal NInfer projector from resident Qwen3.8-27B (5120) -> Qwen-Image 2.1 Qwen3-VL-8B (4096).

Run inside the ComfyUI container:
    docker exec -w /opt/ComfyUI/custom_nodes/comfyui-h3-take -e PYTHONPATH=/opt/ComfyUI \
        comfyui /opt/venv/bin/python -u tools/calibrate_qi21.py
"""

import argparse
import gc
import json
import os
import sys
import time

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ninfer_clip  # noqa: E402
from tools.calibrate_ninfer import (  # noqa: E402
    align_text_spans,
    evaluate_predictions,
    fit_ridge,
    token_char_spans,
    train_residual_mlp,
)


def extract_8b_qi21_targets(clip_name, prompts):
    """Load ComfyUI's Qwen-Image 2.1 Qwen3-VL-8B encoder once and extract last-layer [seq, 4096] targets."""
    import comfy.model_management as mm
    import comfy.sd
    import folder_paths

    path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
    print(f"[QI21-8B] Loading {path} ...", flush=True)
    t0 = time.perf_counter()
    clip = comfy.sd.load_clip(
        ckpt_paths=[path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=comfy.sd.CLIPType.QWEN_IMAGE,
    )
    hf_tok = clip.tokenizer.qwen3vl_8b.tokenizer
    mm.load_models_gpu([clip.patcher])
    print(f"[QI21-8B] Loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    targets = []
    t1 = time.perf_counter()
    try:
        for idx, text in enumerate(prompts):
            tokens = clip.tokenize(text, prevent_empty_text=True)
            raw_ids = [t[0] for t in tokens["qwen3vl_8b"][0] if isinstance(t[0], int)]
            # QwenImage21TEModel drops everything before the 2nd <|im_start|> (151644)
            im_starts = [i for i, tid in enumerate(raw_ids) if tid == 151644]
            drop = im_starts[1] if len(im_starts) > 1 else 0
            kept_ids = raw_ids[drop:]
            tok_texts = [hf_tok.decode([tid], skip_special_tokens=False) for tid in kept_ids]
            with torch.no_grad():
                out = clip.encode_from_tokens(tokens, return_dict=True)
            h = out["cond"][0].to(dtype=torch.bfloat16, device="cpu")  # [seq, 4096]
            if h.shape[0] != len(kept_ids):
                continue
            targets.append({
                "prompt": text,
                "token_texts": tok_texts,
                "spans": token_char_spans(tok_texts),
                "h": h,
            })
            if (idx + 1) % 100 == 0 or idx + 1 == len(prompts):
                print(f"[QI21-8B] Encoded {idx + 1}/{len(prompts)} ({time.perf_counter() - t1:.1f}s)", flush=True)
    finally:
        del clip
        mm.unload_all_models()
        mm.soft_empty_cache()
        gc.collect()
        torch.cuda.empty_cache()
    print(f"[QI21-8B] Unloaded ({len(targets)} prompts captured)", flush=True)
    return targets


def collect_27b_qi21_aligned(url, layer, targets, max_prompts=None):
    """Query NInfer /v1/hidden_states with QI21 system prompt and align to Qwen3-VL-8B kept sequence."""
    subset = targets[:max_prompts] if max_prompts else targets
    x_list, y_list, prompt_ids = [], [], []
    n_exact = n_span = n_unmatched = 0
    t0 = time.perf_counter()

    for p_idx, item in enumerate(subset):
        parts = ninfer_clip.build_parts(item["prompt"])
        meta, h_full = ninfer_clip.fetch_hidden_states(
            url, parts, layer=layer, system_prompt=ninfer_clip.QI21_SYSTEM_PROMPT
        )
        raw_seq = h_full.shape[1]
        tok_texts = meta.get("token_texts") or []
        start, end = ninfer_clip.find_qi21_span(tok_texts, raw_seq)
        u_texts = tok_texts[start:end]
        u_h = h_full[0, start:end]
        src_spans = token_char_spans(u_texts)
        # No attention sink at index 0 here because the system turn absorbed it and was dropped!
        pairs = align_text_spans(src_spans, item["spans"], include_span_mean=True, skip_tgt_zero=False)
        n_unmatched += max(0, len(src_spans) - len(pairs))
        y_h = item["h"].float()
        for src_i, tgt_js, _ in pairs:
            x_list.append(u_h[src_i].to(torch.bfloat16))
            if len(tgt_js) == 1:
                y_list.append(y_h[tgt_js[0]].to(torch.bfloat16))
                n_exact += 1
            else:
                y_list.append(y_h[tgt_js].mean(dim=0).to(torch.bfloat16))
                n_span += 1
            prompt_ids.append(p_idx)

    X = torch.stack(x_list, dim=0)
    Y = torch.stack(y_list, dim=0)
    return X, Y, {
        "n_exact": n_exact,
        "n_span": n_span,
        "n_unmatched": n_unmatched,
        "seconds": round(time.perf_counter() - t0, 2),
        "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long),
    }


def save_qi21_proj(path, ridge, tap, meta_extra, mlp_state=None):
    tensors = {
        "W": ridge["W"].half().contiguous(),
        "mean_in": ridge["mean_in"].half().contiguous(),
        "std_in": ridge["std_in"].half().contiguous(),
        "mean_out": ridge["mean_out"].half().contiguous(),
        "std_out": ridge["std_out"].half().contiguous(),
    }
    if mlp_state is not None:
        for k, v in mlp_state.items():
            tensors["mlp." + k] = v.half().contiguous()
    meta = {
        "target_profile": "qwen_image_21",
        "tap": str(int(tap)),
        "d_in": str(int(ridge["W"].shape[0])),
        "d_out": str(int(ridge["W"].shape[1])),
        "lambda": str(ridge["lambda"]),
        "source_model": "ninfer:qwen3.8-27b",
        "target_model": "qwen3vl_8b_int8_convrot.safetensors",
    }
    meta.update({k: str(v) for k, v in meta_extra.items()})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(tensors, path, metadata=meta)
    print(f"[SAVE] {path} ({os.path.getsize(path) / (1024 * 1024):.1f} MB)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ninfer-url", default=ninfer_clip.DEFAULT_URL)
    parser.add_argument("--clip-8b", default="qwen3vl_8b_int8_convrot.safetensors")
    parser.add_argument("--prompts-json", default="/opt/ComfyUI/output/ninfer_calib/prompts_v2.json")
    parser.add_argument("--n-prompts", type=int, default=990)
    parser.add_argument("--sweep-prompts", type=int, default=120)
    parser.add_argument("--sweep-layers", default="35,43,47,51,55,59,63,-1")
    parser.add_argument("--mlp-hidden", type=int, default=8192)
    parser.add_argument("--mlp-epochs", type=int, default=25)
    parser.add_argument("--out-dir", default="/opt/ComfyUI/models/clip_projections")
    parser.add_argument("--out-prefix", default="qi21-qwen38-27b")
    args = parser.parse_args()

    with open(args.prompts_json, "r", encoding="utf-8") as f:
        loaded = json.load(f)
    prompts = [x["prompt"] if isinstance(x, dict) else str(x) for x in loaded][: args.n_prompts]
    # Also add empty string (negative prompt) and short captions!
    prompts.append("")
    print(f"[CORPUS] Loaded {len(prompts)} prompts", flush=True)

    targets = extract_8b_qi21_targets(args.clip_8b, prompts)

    layers = [int(x.strip()) for x in args.sweep_layers.split(",") if x.strip()]
    print(f"\n=== Qwen-Image 2.1 Layer Sweep ({args.sweep_prompts} prompts, layers={layers}) ===", flush=True)
    n_tr = int(args.sweep_prompts * 0.8)
    lambdas = (100.0, 500.0, 1000.0, 3000.0, 10000.0, 30000.0)
    results = []
    for layer in layers:
        X, Y, st = collect_27b_qi21_aligned(args.ninfer_url, layer, targets, max_prompts=args.sweep_prompts)
        tr = st["prompt_ids"] < n_tr
        res = fit_ridge(X[tr], Y[tr], X[~tr], Y[~tr], lambdas=lambdas)
        results.append((layer, res["cos"], res["r2"], res["lambda"]))
        label = "final" if layer < 0 else f"L{layer:02d}"
        print(
            f"  {label:5s} | cos={res['cos']:.4f} | r2={res['r2']:.4f} | lam={res['lambda']:6.0f} | "
            f"exact={st['n_exact']} span={st['n_span']} unm={st['n_unmatched']} ({st['seconds']}s)",
            flush=True,
        )
    results.sort(key=lambda r: r[1], reverse=True)
    best_layer = results[0][0]
    print(f"--> Best layer for Qwen-Image 2.1: {best_layer} (cos={results[0][1]:.4f}, r2={results[0][2]:.4f})\n", flush=True)

    print(f"=== Full Qwen-Image 2.1 Calibration on layer={best_layer} ({len(targets)} prompts) ===", flush=True)
    X, Y, st = collect_27b_qi21_aligned(args.ninfer_url, best_layer, targets)
    print(f"[27B] Collected in {st['seconds']}s: exact={st['n_exact']} span={st['n_span']} unm={st['n_unmatched']}", flush=True)

    n_tr_full = int(len(targets) * 0.9)
    tr = st["prompt_ids"] < n_tr_full
    ridge = fit_ridge(X[tr], Y[tr], X[~tr], Y[~tr], lambdas=lambdas)
    print(f"[QI21 Ridge] test: cos={ridge['cos']:.4f} r2={ridge['r2']:.4f} (lam={ridge['lambda']})", flush=True)
    save_qi21_proj(
        os.path.join(args.out_dir, f"{args.out_prefix}-ridge.safetensors"),
        ridge,
        best_layer,
        {"n_train_prompts": len(targets), "n_train_tokens": int(tr.sum()), "cos_test": round(ridge["cos"], 6), "r2_test": round(ridge["r2"], 6)},
    )

    print(f"\n=== Training QI21 Residual MLP (5120 -> {args.mlp_hidden} -> 4096, {args.mlp_epochs} epochs) ===", flush=True)
    mlp_state, mlp_cos, mlp_r2 = train_residual_mlp(
        X[tr], Y[tr], X[~tr], Y[~tr], ridge, hidden_dim=args.mlp_hidden, epochs=args.mlp_epochs
    )
    if mlp_state is not None:
        save_qi21_proj(
            os.path.join(args.out_dir, f"{args.out_prefix}-mlp.safetensors"),
            ridge,
            best_layer,
            {
                "n_train_prompts": len(targets),
                "n_train_tokens": int(tr.sum()),
                "cos_test": round(mlp_cos, 6),
                "r2_test": round(mlp_r2, 6),
                "mlp_hidden": args.mlp_hidden,
            },
            mlp_state=mlp_state,
        )
    print("\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()

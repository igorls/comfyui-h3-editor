"""MiniMax H3 v4.1 Audio/Dialogue + Vision Precision Calibrator:
- Multi-layer NInfer tap ([-2, 15, 43, 51, 59] — raw spelling/ViT + mid + best text L51 + best vision L59)
- Lossless UTF-8 byte-level 248k -> 151k target-subword unpacking (F2: 0 U+FFFD drift across Korean/Emoji/PT/CJK)
  with causal last-token N:1 collapse (F4: ['<d','>'] -> last sub-token)
- Zero-VRAM mmapped 32B embed_tokens[tgt_ids] lookup (5120-d)
- Sink-free training filter (||y|| < 2000 drops pos=1 attention sinks on <d>-first prompts so ch 731 never spikes)
  + clean sink_out averaging (||h[0]|| > 2000) + per-channel clamp_lo/hi safety net (1f)
- Two-Block Text Ridge + Dual Multi-Layer Vision Ridge (W_vis)
- 5-Fold Out-of-Fold (OOF) Compact Bidirectional Refinement Transformer with uncompacted ref2va vision-gap
  training sequences (F3) and 3.5x dialogue/lyrics weighting (Huber + cos + rel in standardized space).
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gc
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ninfer_clip  # noqa: E402
import seq_connector  # noqa: E402
from tools.calibrate_ninfer import (  # noqa: E402
    collect_sample_images,
    fit_ridge,
)
from tools.calibrate_qi21_seq import (  # noqa: E402
    compute_gram,
    eval_masked,
    fit_two_block_ridge,
    solve_block_ridge,
)

SINK_NORM_THRESHOLD = 2000.0


def compute_dialogue_mask(tok_texts):
    """Return bool tensor [L] marking tokens inside <d>...</d> or <|lyrics_start|>...<|lyrics_end|>."""
    L = len(tok_texts)
    mask = torch.zeros(L, dtype=torch.bool)
    in_d = False
    in_l = False
    for i, t in enumerate(tok_texts):
        if t == "<d>":
            in_d = True
            mask[i] = True
            continue
        if t == "</d>":
            mask[i] = True
            in_d = False
            continue
        if t == "<|lyrics_start|>":
            in_l = True
            mask[i] = True
            continue
        if t == "<|lyrics_end|>":
            mask[i] = True
            in_l = False
            continue
        if in_d or in_l:
            mask[i] = True
    return mask


def eval_std_r2(y_pred, y_true, m_out, s_out, mask=None):
    """Pooled R² in standardized output space (immune to single-channel variance skew)."""
    if mask is not None:
        if not mask.any():
            return 0.0
        y_pred = y_pred[mask]
        y_true = y_true[mask]
    pn = (y_pred - m_out) / s_out
    yn = (y_true - m_out) / s_out
    num = (pn - yn).pow(2).sum()
    den = (yn - yn.mean(dim=0, keepdim=True)).pow(2).sum().clamp(min=1e-6)
    return 1.0 - (num / den).item()


def flatten_clean_seqs(seqs):
    """Flatten sequences while excluding any attention-sink tokens (non_sink == False)."""
    xs, ys, ds = [], [], []
    for s in seqs:
        m = s["valid"]
        if m.any():
            xs.append(s["x"][m])
            ys.append(s["y"][m])
            ds.append(s["typo"][m])
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0), torch.cat(ds, dim=0)


def extract_32b_targets_with_ids(clip_name, prompts, vision_items):
    """Load MiniMax H3 32B encoder once and extract targets [seq, 5120] + exact token_ids + vision targets."""
    import comfy.model_management as mm
    import comfy.sd
    import folder_paths

    path = folder_paths.get_full_path_or_raise("text_encoders", clip_name)
    print(f"[H3-32B] Loading {path} ...", flush=True)
    t0 = time.perf_counter()
    clip = comfy.sd.load_clip(
        ckpt_paths=[path],
        embedding_directory=folder_paths.get_folder_paths("embeddings"),
        clip_type=comfy.sd.CLIPType.MINIMAX,
    )
    sm = clip.cond_stage_model.qwen3vl_32b
    hf_tok = clip.tokenizer.qwen3vl_32b.tokenizer
    mm.load_models_gpu([clip.patcher])
    print(f"[H3-32B] Loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    captured = {}
    orig_forward = sm.transformer.forward

    def catching(*args, **kwargs):
        captured["embeds_info"] = kwargs.get("embeds_info", [])
        return orig_forward(*args, **kwargs)

    sm.transformer.forward = catching
    text_targets = []
    vision_targets = []
    sinks = []

    try:
        t1 = time.perf_counter()
        for idx, text in enumerate(prompts):
            tokens = clip.tokenize(text)
            raw_ids = [t[0] for t in tokens["qwen3vl_32b"][0] if isinstance(t[0], int)]
            tok_texts = [hf_tok.decode([tid], skip_special_tokens=False) for tid in raw_ids]
            with torch.no_grad():
                out = clip.encode_from_tokens(tokens, return_dict=True)
            h = out["cond"][0].to(dtype=torch.bfloat16, device="cpu")
            if h.shape[0] != len(raw_ids) or h.shape[0] < 2:
                continue
            # Only collect true attention sinks (||h[0]|| > 2000) so <d>-first prompts don't dilute sink_out
            if h[0].float().norm().item() > SINK_NORM_THRESHOLD:
                sinks.append(h[0].float())
            text_targets.append({
                "prompt": text,
                "full_token_ids": raw_ids,
                "full_token_texts": tok_texts,
                "token_ids": raw_ids[1:],
                "token_texts": tok_texts[1:],
                "h": h[1:].contiguous(),
            })
            if (idx + 1) % 200 == 0 or idx + 1 == len(prompts):
                print(f"[H3-32B] Text {idx + 1}/{len(prompts)} ({time.perf_counter() - t1:.1f}s)", flush=True)

        t2 = time.perf_counter()
        for idx, (img_name, img_tensor, prompt_text) in enumerate(vision_items):
            captured.clear()
            tokens = clip.tokenize(prompt_text, images=[img_tensor])
            raw_entries = tokens["qwen3vl_32b"][0]
            # Split raw entries into prefix int ids (before VISION_START) and suffix int ids (after VISION_END)
            img_pos = next((k for k, e in enumerate(raw_entries) if not isinstance(e[0], int)), None)
            with torch.no_grad():
                out = clip.encode_from_tokens(tokens, return_dict=True)
            h = out["cond"][0].to(dtype=torch.bfloat16, device="cpu")
            embeds_info = captured.get("embeds_info", [])
            img_span = None
            for e in embeds_info:
                if e.get("type") == "image":
                    img_span = (int(e["index"]), int(e["size"]))
                    break
            if img_span is not None and img_pos is not None:
                v0 = max(0, img_span[0] - 1)
                v1 = min(h.shape[0], img_span[0] + img_span[1] + 1)
                prefix_ids = [raw_entries[k][0] for k in range(max(0, img_pos - 1)) if isinstance(raw_entries[k][0], int)]
                suffix_ids = [raw_entries[k][0] for k in range(min(len(raw_entries), img_pos + 2), len(raw_entries)) if isinstance(raw_entries[k][0], int)]
                vision_targets.append({
                    "name": img_name,
                    "prompt": prompt_text,
                    "img_tensor": img_tensor,
                    "vis_size": img_span[1],
                    "vis_block_len": v1 - v0,
                    "v0": v0,
                    "v1": v1,
                    "prefix_ids": prefix_ids,
                    "suffix_ids": suffix_ids,
                    "prefix_texts": [hf_tok.decode([tid], skip_special_tokens=False) for tid in prefix_ids],
                    "suffix_texts": [hf_tok.decode([tid], skip_special_tokens=False) for tid in suffix_ids],
                    "h_vis": h[v0:v1].contiguous(),
                    "h_full": h.contiguous(),
                })
        print(f"[H3-32B] Vision {len(vision_targets)}/{len(vision_items)} ({time.perf_counter() - t2:.1f}s)", flush=True)
    finally:
        sm.transformer.forward = orig_forward
        del clip
        mm.unload_all_models()
        mm.soft_empty_cache()
        gc.collect()
        torch.cuda.empty_cache()

    sink_out = torch.stack(sinks, dim=0).mean(dim=0)
    print(f"[H3-32B] Unloaded. sink_out norm={sink_out.norm().item():.2f} (from {len(sinks)} clean sinks)", flush=True)
    return text_targets, vision_targets, sink_out


def collect_h3_multilayer(url, layers, text_targets, vision_targets, clip_32b_name=None, workers=4):
    """Fetch multi-layer 27B states, unpack 248k -> 151k MiniMax 32B text tokens via lossless UTF-8
    byte spans + causal last-token N:1 + 0-VRAM mmapped 32B embed_tokens; also collect 1:1 vision blocks
    AND full uncompacted ref2va text-with-vision-gap sequences for F3 Seq training."""
    t0 = time.perf_counter()
    tgt_tok = ninfer_clip.get_target_tokenizer("minimax_h3")
    if clip_32b_name:
        ninfer_clip.lookup_target_embeddings(clip_32b_name, [0])

    byte_mismatches = [0]

    def fetch_text(p_idx):
        item = text_targets[p_idx]
        parts = ninfer_clip.build_parts(item["prompt"])
        meta, h_full = ninfer_clip.fetch_hidden_states(url, parts, layers=layers)
        raw_seq = h_full.shape[1]
        tok_texts = meta.get("token_texts") or []
        tok_blens = meta.get("token_byte_lens")
        start, end = ninfer_clip.find_user_span(tok_texts, raw_seq)
        u_h = h_full[0, start:end]
        if isinstance(tok_blens, list) and len(tok_blens) == raw_seq:
            src_blens = tok_blens[start:end]
            tgt_blens = ninfer_clip.target_token_byte_lens(tgt_tok, item["full_token_ids"])
            if sum(src_blens) != sum(tgt_blens):
                byte_mismatches[0] += 1
            unpacked = ninfer_clip.unpack_by_spans(u_h, src_blens, tgt_blens, n_to_1="last")[1:]
        else:
            u_texts = tok_texts[start:end]
            unpacked = ninfer_clip.unpack_by_char_spans(u_h, u_texts, item["full_token_texts"], n_to_1="last")[1:]
        if clip_32b_name:
            emb = ninfer_clip.lookup_target_embeddings(clip_32b_name, item["token_ids"])
            x = torch.cat([unpacked, emb], dim=-1)
        else:
            x = unpacked
        y = item["h"].contiguous()
        non_sink = y.float().norm(dim=-1) < SINK_NORM_THRESHOLD
        return {
            "x": x.to(torch.bfloat16).contiguous(),
            "y": y,
            "typo": compute_dialogue_mask(item["token_texts"]),
            "valid": non_sink,
            "prompt_idx": p_idx,
        }

    text_seqs = [None] * len(text_targets)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(fetch_text, range(len(text_targets))):
            text_seqs[res["prompt_idx"]] = res
            done += 1
            if done % 250 == 0 or done == len(text_targets):
                print(f"[27B-TXT] Collected {done}/{len(text_targets)} ({time.perf_counter() - t0:.1f}s)", flush=True)
    n_sinks_dropped = sum(int((~s["valid"]).sum().item()) for s in text_seqs)
    print(f"[27B-TXT] Byte-span mismatches: {byte_mismatches[0]}/{len(text_targets)} | pos>=1 sinks masked: {n_sinks_dropped}", flush=True)

    # Vision collection (sequential to avoid overloading vision tower) + full uncompacted ref2va seqs (F3)
    x_vis_list, y_vis_list, vis_img_ids = [], [], []
    ref_text_seqs = []
    matched_imgs = 0
    t1 = time.perf_counter()
    d_vis = len(layers) * 5120
    d_txt = d_vis + (5120 if clip_32b_name else 0)
    for v_idx, item in enumerate(vision_targets):
        parts = ninfer_clip.build_parts(item["prompt"], images=[item["img_tensor"]])
        meta, h_full = ninfer_clip.fetch_hidden_states(url, parts, layers=layers)
        raw_seq = h_full.shape[1]
        t_types = meta.get("token_types") or []
        tok_texts = meta.get("token_texts") or []
        tok_blens = meta.get("token_byte_lens")
        start, end = ninfer_clip.find_user_span(tok_texts, raw_seq)
        vis_pad = [i for i, t in enumerate(t_types) if int(t) != 0]
        if len(vis_pad) == item["vis_size"] and vis_pad:
            s0 = vis_pad[0] - 1 if vis_pad[0] > 0 and (not tok_texts or tok_texts[vis_pad[0] - 1] == "<|vision_start|>") else vis_pad[0]
            s1 = vis_pad[-1] + 2 if vis_pad[-1] + 1 < raw_seq and (not tok_texts or tok_texts[vis_pad[-1] + 1] == "<|vision_end|>") else vis_pad[-1] + 1
            if (s1 - s0) == item["vis_block_len"]:
                x_vis_list.append(h_full[0, s0:s1].to(torch.bfloat16))
                y_vis_list.append(item["h_vis"].to(torch.bfloat16))
                vis_img_ids.extend([v_idx] * (s1 - s0))
                matched_imgs += 1

                # Also construct uncompacted ref2va text-with-vision-gap sequence (excluding token 0 sink) for F3!
                use_bytes = isinstance(tok_blens, list) and len(tok_blens) == raw_seq
                p_ids, suf_ids = item["prefix_ids"], item["suffix_ids"]
                if len(p_ids) == item["v0"] and len(suf_ids) == (item["h_full"].shape[0] - item["v1"]) and s0 > start and end >= s1:
                    if use_bytes:
                        p_u = ninfer_clip.unpack_by_spans(
                            h_full[0, start:s0], tok_blens[start:s0],
                            ninfer_clip.target_token_byte_lens(tgt_tok, p_ids), n_to_1="last"
                        )
                        s_u = ninfer_clip.unpack_by_spans(
                            h_full[0, s1:end], tok_blens[s1:end],
                            ninfer_clip.target_token_byte_lens(tgt_tok, suf_ids), n_to_1="last"
                        )
                    else:
                        p_u = ninfer_clip.unpack_by_char_spans(h_full[0, start:s0], tok_texts[start:s0], item["prefix_texts"], n_to_1="last")
                        s_u = ninfer_clip.unpack_by_char_spans(h_full[0, s1:end], tok_texts[s1:end], item["suffix_texts"], n_to_1="last")
                    if clip_32b_name:
                        p_u = torch.cat([p_u, ninfer_clip.lookup_target_embeddings(clip_32b_name, p_ids)], dim=-1)
                        s_u = torch.cat([s_u, ninfer_clip.lookup_target_embeddings(clip_32b_name, suf_ids)], dim=-1)
                    L_full = item["h_full"].shape[0]
                    x_ref = torch.zeros(L_full, d_txt, dtype=torch.bfloat16)
                    x_ref[: item["v0"]] = p_u.to(torch.bfloat16)
                    x_ref[item["v1"] :] = s_u.to(torch.bfloat16)
                    valid_ref = torch.zeros(L_full, dtype=torch.bool)
                    valid_ref[: item["v0"]] = True
                    valid_ref[item["v1"] :] = True
                    # Drop token 0 (sink) and mask any residual >2000 sink
                    y_sub = item["h_full"][1:].contiguous()
                    v_sub = valid_ref[1:] & (y_sub.float().norm(dim=-1) < SINK_NORM_THRESHOLD)
                    d_sub = torch.zeros(L_full - 1, dtype=torch.bool)
                    if item["v1"] - 1 < L_full - 1:
                        d_sub[item["v1"] - 1 :] = compute_dialogue_mask(item["suffix_texts"])
                    ref_text_seqs.append({
                        "x": x_ref[1:].contiguous(),
                        "y": y_sub,
                        "typo": d_sub,
                        "valid": v_sub,
                        "prompt_idx": -1 - v_idx,
                    })

    X_vis = torch.cat(x_vis_list, dim=0) if x_vis_list else torch.empty(0, d_vis, dtype=torch.bfloat16)
    Y_vis = torch.cat(y_vis_list, dim=0) if y_vis_list else torch.empty(0, 5120, dtype=torch.bfloat16)
    vis_ids = torch.tensor(vis_img_ids, dtype=torch.long)
    print(f"[27B-VIS] Matched {matched_imgs}/{len(vision_targets)} imgs ({X_vis.shape[0]} toks, {len(ref_text_seqs)} F3 ref-gap seqs) in {time.perf_counter()-t1:.1f}s", flush=True)
    return text_seqs, (X_vis, Y_vis, vis_ids, matched_imgs), ref_text_seqs


def compute_oof_bases_clean(tr_seqs, te_seqs, ref_seqs, ridge, d_main, n_folds=5, device="cuda"):
    """5-fold OOF Ridge predictions with strict non-sink masking inside every fold."""
    t0 = time.perf_counter()
    lm, le = ridge["lam_main"], ridge["lam_emb"]
    m_out_full = ridge["mean_out"].to(device)
    s_out_full = ridge["std_out"].to(device)

    n = len(tr_seqs)
    fold_size = (n + n_folds - 1) // n_folds
    oof_bases = [None] * n

    for f in range(n_folds):
        v0 = f * fold_size
        v1 = min(n, v0 + fold_size)
        val_s = tr_seqs[v0:v1]
        trn_s = tr_seqs[:v0] + tr_seqs[v1:]
        Xtr, Ytr, _ = flatten_clean_seqs(trn_s)
        xtx, xty, m_in, s_in, m_out, s_out = compute_gram(Xtr, Ytr, device=device)
        W_f = solve_block_ridge(xtx, xty, d_main, lm, le, device=device)
        mi_g, si_g = m_in.to(device), s_in.to(device)
        mo_g, so_g = m_out.to(device), s_out.to(device)
        with torch.no_grad():
            for offset, s in enumerate(val_s):
                xn = (s["x"].float().to(device) - mi_g) / si_g
                pred_raw = (xn @ W_f) * so_g + mo_g
                pred_n = (pred_raw - m_out_full) / s_out_full
                oof_bases[v0 + offset] = pred_n.to(torch.bfloat16).cpu()
        del Xtr, Ytr, xtx, xty, W_f
        torch.cuda.empty_cache()

    mi_g, si_g = ridge["mean_in"].to(device), ridge["std_in"].to(device)
    W_g = ridge["W"].to(device)
    te_bases = []
    ref_bases = []
    with torch.no_grad():
        for s in te_seqs:
            xn = (s["x"].float().to(device) - mi_g) / si_g
            te_bases.append((xn @ W_g).to(torch.bfloat16).cpu())
        for s in ref_seqs:
            xn = (s["x"].float().to(device) - mi_g) / si_g
            bn = xn @ W_g
            bn[~s["valid"].to(device)] = 0.0
            ref_bases.append(bn.to(torch.bfloat16).cpu())

    oof_p = torch.cat([((b.float().to(device) * s_out_full + m_out_full)[s["valid"].to(device)]).cpu() for b, s in zip(oof_bases, tr_seqs)], 0)
    oof_y = torch.cat([s["y"][s["valid"]].float() for s in tr_seqs], 0)
    oof_t = torch.cat([s["typo"][s["valid"]] for s in tr_seqs], 0)
    ac, ar, _, _ = eval_masked(oof_p, oof_y)
    dc, dr, _, _ = eval_masked(oof_p, oof_y, oof_t)
    print(f"[OOF {n_folds}-fold] {time.perf_counter()-t0:.1f}s | OOF ALL cos={ac:.4f} R2={ar:.4f} | OOF DIAL cos={dc:.4f} R2={dr:.4f}", flush=True)
    return oof_bases, te_bases, ref_bases


def train_oof_seq_clean(
    tr_seqs, te_seqs, oof_bases, te_bases, ridge,
    ref_seqs=(), ref_bases=(),
    d_model=768, n_layers=2, n_heads=8, epochs=40, batch_size=32, lr=4e-4, dial_weight=3.5, device="cuda",
):
    """Train Compact OOF Bidirectional Refinement Transformer with clean non-sink masking,
    uncompacted ref2va gap sequences (F3), Huber+cos+rel loss in standardized space, and
    checkpoint selection on DIAL std_R2 + cos."""
    d_out = ridge["W"].shape[1]
    m_out = ridge["mean_out"].to(device)
    s_out = ridge["std_out"].to(device)

    all_tr_seqs = list(tr_seqs) + list(ref_seqs)
    all_tr_bases = list(oof_bases) + list(ref_bases)

    net = seq_connector.BidirectionalSeqConnector(
        d_in=d_out, d_out=d_out, d_model=d_model, n_layers=n_layers, n_heads=n_heads,
        ffn_mult=2, conv_kernel=5, dropout=0.15,
    ).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[REFINE-SEQ] Params: {n_params/1e6:.1f}M ({d_out} -> d_model={d_model} x {n_layers}L -> {d_out}) | train_seqs={len(all_tr_seqs)} (incl {len(ref_seqs)} ref-gap)", flush=True)

    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.03)
    warmup = 3

    def lr_lambda(ep):
        if ep < warmup:
            return float(ep + 1) / float(warmup)
        prog = float(ep - warmup) / max(1.0, float(epochs - warmup))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    def make_batch(seqs, bases, idxs):
        B = len(idxs)
        Lmax = max(seqs[k]["y"].shape[0] for k in idxs)
        Base = torch.zeros(B, Lmax, d_out, dtype=torch.bfloat16)
        Y = torch.zeros(B, Lmax, d_out, dtype=torch.bfloat16)
        valid = torch.zeros(B, Lmax, dtype=torch.bool)
        typo = torch.zeros(B, Lmax, dtype=torch.bool)
        for i, k in enumerate(idxs):
            L = seqs[k]["y"].shape[0]
            vm = seqs[k]["valid"]
            Base[i, :L] = bases[k]
            Base[i, :L][~vm] = 0
            Y[i, :L] = seqs[k]["y"]
            valid[i, :L] = vm
            typo[i, :L] = seqs[k]["typo"]
        return (
            Base.to(device, non_blocking=True),
            Y.to(device, non_blocking=True),
            valid.to(device, non_blocking=True),
            typo.to(device, non_blocking=True),
        )

    def evaluate():
        net.eval()
        ps, ts, tm = [], [], []
        with torch.no_grad():
            for i in range(0, len(te_seqs), batch_size):
                idxs = list(range(i, min(len(te_seqs), i + batch_size)))
                Base, Y, valid, typo = make_batch(te_seqs, te_bases, idxs)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    delta = net(Base, valid_mask=valid)
                pred = (Base.float() + delta.float()) * s_out + m_out
                ps.append(pred[valid])
                ts.append(Y.float()[valid])
                tm.append(typo[valid])
        P, T, M = torch.cat(ps, 0), torch.cat(ts, 0), torch.cat(tm, 0)
        ac, ar, arel, _ = eval_masked(P, T)
        dc, dr, drel, dnorm = eval_masked(P, T, M)
        a_sr2 = eval_std_r2(P, T, m_out, s_out)
        d_sr2 = eval_std_r2(P, T, m_out, s_out, M)
        max_r = (P.norm(dim=-1) / T.norm(dim=-1).clamp(min=1e-6)).max().item()
        return ac, ar, arel, a_sr2, dc, dr, drel, dnorm, d_sr2, max_r

    b = evaluate()
    print(
        f"  [REFINE] ep  0/{epochs} (Ridge) | ALL cos={b[0]:.4f} R2={b[1]:.4f} stdR2={b[3]:.4f} | "
        f"DIAL cos={b[4]:.4f} R2={b[5]:.4f} stdR2={b[8]:.4f} norm={b[7]:.3f} max={b[9]:.2f}x",
        flush=True,
    )
    best_score = 0.25 * b[0] + 0.25 * b[3] + 0.25 * b[4] + 0.25 * b[8]
    best_metrics = b
    best_state = None
    rng = random.Random(42)
    t0 = time.perf_counter()

    for ep in range(epochs):
        net.train()
        order = list(range(len(all_tr_seqs)))
        rng.shuffle(order)
        for i in range(0, len(order), batch_size):
            Base, Y, valid, typo = make_batch(all_tr_seqs, all_tr_bases, order[i : i + batch_size])
            yt = Y.float()
            yn_true = (yt - m_out) / s_out
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                delta = net(Base, valid_mask=valid)
                pred_n = Base.float() + delta.float()
                pred = pred_n * s_out + m_out
                pn_a, yn_a = pred_n[valid], yn_true[valid]
                p_a, y_a = pred[valid], yt[valid]
                w = torch.where(typo[valid], dial_weight, 1.0)
                w = w / w.mean().clamp(min=1e-6)
                tok_huber = F.smooth_l1_loss(pn_a, yn_a, reduction="none", beta=1.5).mean(dim=-1)
                tok_cos = 1.0 - F.cosine_similarity(p_a, y_a, dim=-1)
                tok_rel = (p_a - y_a).norm(dim=-1) / y_a.norm(dim=-1).clamp(min=1e-6)
                loss = (w * (tok_huber + 1.0 * tok_cos + 0.35 * tok_rel)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
        sched.step()

        cur = evaluate()
        score = 0.25 * cur[0] + 0.25 * cur[3] + 0.25 * cur[4] + 0.25 * cur[8]
        if score > best_score:
            best_score = score
            best_metrics = cur
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        if (ep + 1) % 5 == 0 or ep + 1 == epochs or ep == 0:
            print(
                f"  [REFINE] ep {ep + 1:2d}/{epochs} | ALL cos={cur[0]:.4f} R2={cur[1]:.4f} stdR2={cur[3]:.4f} | "
                f"DIAL cos={cur[4]:.4f} R2={cur[5]:.4f} stdR2={cur[8]:.4f} norm={cur[7]:.3f} max={cur[9]:.2f}x ({time.perf_counter()-t0:.1f}s)",
                flush=True,
            )

    if best_state is None:
        return None, best_metrics
    net.load_state_dict(best_state)
    return net, best_metrics


def save_h3_v4_proj(path, ridge_txt, ridge_vis, sink_out, clamp_lo, clamp_hi, layers, clip_32b_name, meta_extra, net=None):
    tensors = {
        "W": ridge_txt["W"].half().contiguous(),
        "mean_in": ridge_txt["mean_in"].half().contiguous(),
        "std_in": ridge_txt["std_in"].half().contiguous(),
        "mean_out": ridge_txt["mean_out"].half().contiguous(),
        "std_out": ridge_txt["std_out"].half().contiguous(),
        "sink_out": sink_out.half().contiguous(),
        "clamp_lo": clamp_lo.half().contiguous(),
        "clamp_hi": clamp_hi.half().contiguous(),
    }
    if ridge_vis is not None:
        tensors["W_vis"] = ridge_vis["W"].half().contiguous()
        tensors["mean_in_vis"] = ridge_vis["mean_in"].half().contiguous()
        tensors["std_in_vis"] = ridge_vis["std_in"].half().contiguous()
        tensors["mean_out_vis"] = ridge_vis["mean_out"].half().contiguous()
        tensors["std_out_vis"] = ridge_vis["std_out"].half().contiguous()
    meta = {
        "target_profile": "minimax_h3",
        "unpack_target_vocab": "minimax_h3",
        "unpack_n_to_1": "last",
        "target_embed_model": clip_32b_name or "",
        "tap": str(int(layers[-1])),
        "layers": json.dumps([int(x) for x in layers]),
        "d_in": str(int(ridge_txt["W"].shape[0])),
        "d_out": str(int(ridge_txt["W"].shape[1])),
        "lambda": str(ridge_txt["lam_main"]),
        "lambda_emb": str(ridge_txt["lam_emb"]),
        "source_model": "ninfer:qwen3.8-27b",
        "target_model": clip_32b_name or "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    }
    if net is not None:
        tensors.update(seq_connector.state_dict_prefixed(net, prefix="seq.", dtype=torch.bfloat16))
        meta["connector_type"] = "seq_base_refine_v1"
        meta["seq_input"] = "base"
        meta["seq_config"] = json.dumps(net.config)
    meta.update({k: str(v) for k, v in meta_extra.items()})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_file(tensors, path, metadata=meta)
    print(f"[SAVE] {path} ({os.path.getsize(path) / (1024 * 1024):.1f} MB)", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ninfer-url", default=ninfer_clip.DEFAULT_URL)
    parser.add_argument("--clip-32b", default="qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors")
    parser.add_argument("--prompts-json", default="/content/work/prompts/prompts_h3_v4.json")
    parser.add_argument("--n-prompts", type=int, default=2050)
    parser.add_argument("--n-vision", type=int, default=80)
    parser.add_argument("--vision-dir", default="/content/work/vision_samples")
    parser.add_argument("--layers", default="-2,15,43,51,59")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--d-model", type=int, default=768)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--n-heads", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dial-weight", type=float, default=3.5)
    parser.add_argument("--cache-path", default="/content/h3_v41_cache.pt")
    parser.add_argument("--out-dir", default="/content/ComfyUI/models/clip_projections")
    parser.add_argument("--out-prefix", default="mmh3-qwen38-27b-v4.1")
    args = parser.parse_args()

    layers = [int(x.strip()) for x in args.layers.split(",") if x.strip()]
    d_main = len(layers) * 5120

    if args.cache_path and os.path.isfile(args.cache_path):
        print(f"[CACHE] Loading {args.cache_path} ...", flush=True)
        blob = torch.load(args.cache_path, weights_only=False)
        text_seqs = blob["text_seqs"]
        X_vis, Y_vis, vis_ids, matched_imgs = blob["vis"]
        ref_text_seqs = blob.get("ref_text_seqs", [])
        sink_out = blob["sink_out"]
    else:
        with open(args.prompts_json, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        prompts = [x["prompt"] if isinstance(x, dict) else str(x) for x in loaded]
        random.Random(42).shuffle(prompts)
        prompts = prompts[: args.n_prompts]
        print(f"[CORPUS] Loaded {len(prompts)} shuffled prompts from {args.prompts_json}", flush=True)

        raw_imgs = collect_sample_images(args.n_vision, search_dir=args.vision_dir)
        vision_items = [(name, img, prompts[i % len(prompts)]) for i, (name, img) in enumerate(raw_imgs)]
        text_targets, vision_targets, sink_out = extract_32b_targets_with_ids(args.clip_32b, prompts, vision_items)
        text_seqs, (X_vis, Y_vis, vis_ids, matched_imgs), ref_text_seqs = collect_h3_multilayer(
            args.ninfer_url, layers, text_targets, vision_targets,
            clip_32b_name=args.clip_32b, workers=args.workers
        )
        if args.cache_path:
            torch.save({
                "text_seqs": text_seqs,
                "vis": (X_vis, Y_vis, vis_ids, matched_imgs),
                "ref_text_seqs": ref_text_seqs,
                "sink_out": sink_out,
            }, args.cache_path)
            print(f"[CACHE] Saved {args.cache_path}", flush=True)

    n_tr = int(len(text_seqs) * 0.9)
    tr_seqs, te_seqs = text_seqs[:n_tr], text_seqs[n_tr:]
    Xtr, Ytr, _ = flatten_clean_seqs(tr_seqs)
    Xte, Yte, Dte = flatten_clean_seqs(te_seqs)
    print(f"[SPLIT] train={len(tr_seqs)} seqs ({Xtr.shape[0]} clean toks) | test={len(te_seqs)} seqs ({Xte.shape[0]} clean toks, {int(Dte.sum())} dial) | ref_gap={len(ref_text_seqs)} | d_in={Xtr.shape[1]}", flush=True)

    ridge_txt = fit_two_block_ridge(
        Xtr, Ytr, Xte, Yte, Dte, d_main=d_main,
        lam_mains=(10000.0, 30000.0, 100000.0),
        lam_embs=(3000.0, 10000.0, 30000.0),
    )
    W_diag = ridge_txt["W"]
    blk_names = [f"L{l}" for l in layers] + (["e_tgt"] if W_diag.shape[0] > d_main else [])
    blk_norms = [f"{nm}={W_diag[i*5120:(i+1)*5120].norm().item():.1f}" for i, nm in enumerate(blk_names)]
    print("[W-BLOCK-NORMS] " + " | ".join(blk_norms), flush=True)

    # Clean per-channel clamp safety net (1f)
    Ytr_f = Ytr.float()
    clamp_lo = Ytr_f.min(dim=0).values - 2.0 * ridge_txt["std_out"]
    clamp_hi = Ytr_f.max(dim=0).values + 2.0 * ridge_txt["std_out"]
    del Ytr_f

    # Check SVD compression on the CLEAN W (without ch 731 poisoning the spectrum!)
    with torch.no_grad():
        device = "cuda"
        U, S, Vh = torch.linalg.svd(ridge_txt["W"].float().to(device), full_matrices=False)
        Xte_n = (Xte.float().to(device) - ridge_txt["mean_in"].to(device)) / ridge_txt["std_in"].to(device)
        Yte_g = Yte.float().to(device)
        Dte_g = Dte.to(device)
        mo_g, so_g = ridge_txt["mean_out"].to(device), ridge_txt["std_out"].to(device)
        print("[CLEAN SVD CHECK]", flush=True)
        for rank in (1024, 1536, 2048):
            Wr = (U[:, :rank] * S[:rank]) @ Vh[:rank, :]
            Pr = (Xte_n @ Wr) * so_g + mo_g
            ac, ar, _, _ = eval_masked(Pr, Yte_g)
            dc, dr, _, _ = eval_masked(Pr, Yte_g, Dte_g)
            print(f"  rank={rank:4d}: ALL cos={ac:.4f} R2={ar:.4f} | DIAL cos={dc:.4f} R2={dr:.4f}", flush=True)
        del U, S, Vh, Xte_n, Yte_g, Dte_g

    ridge_vis = None
    if X_vis.shape[0] > 0:
        n_v_tr = max(1, int(matched_imgs * 0.85))
        tr_v = vis_ids < n_v_tr
        te_v = ~tr_v
        if tr_v.any() and te_v.any():
            ridge_vis = fit_ridge(
                X_vis[tr_v], Y_vis[tr_v], X_vis[te_v], Y_vis[te_v],
                lambdas=(100000.0, 300000.0, 1000000.0, 3000000.0),
            )
            print(f"[Dual Multi-Layer Vision Ridge] cos={ridge_vis['cos']:.4f} r2={ridge_vis['r2']:.4f} (lam={ridge_vis['lambda']:.0f})", flush=True)

    save_h3_v4_proj(
        os.path.join(args.out_dir, f"{args.out_prefix}-dual-ridge.safetensors"),
        ridge_txt, ridge_vis, sink_out, clamp_lo, clamp_hi, layers, args.clip_32b,
        {
            "n_train_prompts": len(tr_seqs),
            "n_train_tokens": int(Xtr.shape[0]),
            "cos_test": round(ridge_txt["cos"], 6),
            "r2_test": round(ridge_txt["r2"], 6),
            "dial_cos_test": round(ridge_txt["typo_cos"], 6),
            "dial_r2_test": round(ridge_txt["typo_r2"], 6),
            "cos_test_vis": round(ridge_vis["cos"], 6) if ridge_vis else 0.0,
            "r2_test_vis": round(ridge_vis["r2"], 6) if ridge_vis else 0.0,
        },
        net=None,
    )
    del Xtr, Ytr, Xte, Yte
    gc.collect()
    torch.cuda.empty_cache()

    print("\n=== Computing 5-Fold Out-of-Fold Clean Ridge Bases ===", flush=True)
    oof_bases, te_bases, ref_bases = compute_oof_bases_clean(
        tr_seqs, te_seqs, ref_text_seqs, ridge_txt, d_main=d_main, n_folds=5
    )

    print(f"\n=== Training Clean Compact OOF Bidirectional Refinement Connector ({args.epochs} epochs) ===", flush=True)
    net, best_m = train_oof_seq_clean(
        tr_seqs, te_seqs, oof_bases, te_bases, ridge_txt,
        ref_seqs=ref_text_seqs, ref_bases=ref_bases,
        d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads,
        epochs=args.epochs, batch_size=args.batch_size, dial_weight=args.dial_weight,
    )
    if net is not None:
        save_h3_v4_proj(
            os.path.join(args.out_dir, f"{args.out_prefix}-dual-seq.safetensors"),
            ridge_txt, ridge_vis, sink_out, clamp_lo, clamp_hi, layers, args.clip_32b,
            {
                "n_train_prompts": len(tr_seqs),
                "n_train_tokens": int(sum(int(s["valid"].sum().item()) for s in tr_seqs)),
                "cos_test": round(best_m[0], 6),
                "r2_test": round(best_m[1], 6),
                "std_r2_test": round(best_m[3], 6),
                "dial_cos_test": round(best_m[4], 6),
                "dial_r2_test": round(best_m[5], 6),
                "dial_std_r2_test": round(best_m[8], 6),
                "max_norm_ratio_test": round(best_m[9], 4),
                "cos_test_vis": round(ridge_vis["cos"], 6) if ridge_vis else 0.0,
                "r2_test_vis": round(ridge_vis["r2"], 6) if ridge_vis else 0.0,
            },
            net=net,
        )
    print("\n=== DONE ===", flush=True)


if __name__ == "__main__":
    main()

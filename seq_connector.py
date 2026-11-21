"""Bidirectional multi-layer sequence connector for mapping external LLM hidden states
(e.g. Qwen3.8-27B multi-layer taps from NInfer) into diffusion encoder spaces (Qwen-Image 2.1,
MiniMax H3) with high typography/spelling and spoken phoneme precision.

Applied as a zero-initialized residual on top of closed-form Multi-Layer Ridge in standardized space:
    xn = (h - mean_in) / std_in                 # [B, seq, K * d_base]
    yn = xn @ W + seq_connector(xn, mask)       # [B, seq, d_out]
    cond = yn * std_out + mean_out
"""

import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _rope_cos_sin(seq_len, head_dim, device, dtype, theta=10000.0):
    half = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, device=device, dtype=torch.float32) / half))
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(pos, inv_freq)  # [L, half]
    cos = freqs.cos().unsqueeze(0).unsqueeze(0).to(dtype)  # [1, 1, L, half]
    sin = freqs.sin().unsqueeze(0).unsqueeze(0).to(dtype)
    return cos, sin


def _apply_rope(x, cos, sin):
    # x: [B, H, L, D]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class SeqConnectorBlock(nn.Module):
    """Pre-norm block combining:
    1. Local bidirectional depthwise Conv1D (k=5) for adjacent subword / spelling / phoneme fusion
    2. Bidirectional multi-head self-attention with 1D RoPE for full-prompt context (quotes, <d> tags)
    3. SwiGLU FFN
    """

    def __init__(self, d_model, n_heads, ffn_dim, conv_kernel=5, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert self.head_dim * n_heads == d_model and self.head_dim % 2 == 0

        # 1. Local bidirectional subword conv branch
        self.conv_norm = nn.LayerNorm(d_model)
        self.dw_conv = nn.Conv1d(
            d_model, d_model, kernel_size=conv_kernel, padding=conv_kernel // 2, groups=d_model
        )
        self.conv_out = nn.Linear(d_model, d_model)

        # 2. Bidirectional RoPE self-attention
        self.attn_norm = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # 3. SwiGLU FFN
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, 2 * ffn_dim, bias=False)
        self.ffn_out = nn.Linear(ffn_dim, d_model, bias=False)

        self.drop = nn.Dropout(dropout)
        self.attn_drop_p = float(dropout)

    def forward(self, x, cos, sin, valid_mask=None, attn_mask=None):
        # valid_mask: [B, L, 1] float (1 for valid, 0 for pad) or None
        # attn_mask:  [B, 1, 1, L] bool (True = attend, False = pad) or None

        # Local Conv1D residual
        h = self.conv_norm(x)
        if valid_mask is not None:
            h = h * valid_mask
        c = self.dw_conv(h.transpose(1, 2)).transpose(1, 2)
        c = self.conv_out(F.silu(c))
        if valid_mask is not None:
            c = c * valid_mask
        x = x + self.drop(c)

        # Bidirectional RoPE self-attention residual
        B, L, D = x.shape
        h = self.attn_norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = q.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        a = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.attn_drop_p if self.training else 0.0,
            is_causal=False,
        )
        a = a.transpose(1, 2).contiguous().view(B, L, D)
        a = self.o_proj(a)
        if valid_mask is not None:
            a = a * valid_mask
        x = x + self.drop(a)

        # SwiGLU FFN residual
        h = self.ffn_norm(x)
        gate, up = self.ffn_in(h).chunk(2, dim=-1)
        f = self.ffn_out(F.silu(gate) * up)
        if valid_mask is not None:
            f = f * valid_mask
        x = x + self.drop(f)
        return x


class BidirectionalSeqConnector(nn.Module):
    """Zero-initialized residual sequence connector: [B, L, d_in] -> [B, L, d_out]."""

    def __init__(
        self,
        d_in,
        d_out,
        d_model=1536,
        n_layers=3,
        n_heads=12,
        ffn_mult=2,
        conv_kernel=5,
        dropout=0.1,
    ):
        super().__init__()
        self.config = {
            "d_in": int(d_in),
            "d_out": int(d_out),
            "d_model": int(d_model),
            "n_layers": int(n_layers),
            "n_heads": int(n_heads),
            "ffn_mult": int(ffn_mult),
            "conv_kernel": int(conv_kernel),
        }
        ffn_dim = int(d_model * ffn_mult)
        self.in_drop = nn.Dropout(dropout)
        self.in_proj = nn.Linear(d_in, d_model)
        self.in_norm = nn.LayerNorm(d_model)
        self.blocks = nn.ModuleList(
            [
                SeqConnectorBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ffn_dim=ffn_dim,
                    conv_kernel=conv_kernel,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.out_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, d_out)
        # Zero-init output projection so epoch 0 starts cleanly at the Ridge solution.
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, xn, valid_mask=None):
        """
        xn:         [B, L, d_in] standardized multi-layer hidden states
        valid_mask: optional [B, L] bool (True = valid token, False = pad)
        returns:    [B, L, d_out] residual in standardized output space
        """
        B, L, _ = xn.shape
        v_mask = valid_mask.unsqueeze(-1).to(xn.dtype) if valid_mask is not None else None
        a_mask = valid_mask.unsqueeze(1).unsqueeze(2) if valid_mask is not None else None

        x = self.in_norm(self.in_proj(self.in_drop(xn)))
        if v_mask is not None:
            x = x * v_mask

        head_dim = self.blocks[0].head_dim
        cos, sin = _rope_cos_sin(L, head_dim, x.device, x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin, valid_mask=v_mask, attn_mask=a_mask)

        out = self.out_proj(self.out_norm(x))
        if v_mask is not None:
            out = out * v_mask
        return out


def build_from_projection(proj, device, prefix="seq."):
    """Instantiate and load a BidirectionalSeqConnector from a clipproj dict, or return None."""
    seq_keys = [k for k in proj if k.startswith(prefix)]
    if not seq_keys:
        return None
    cfg_key = "seq_config" if prefix == "seq." else "seq_vis_config"
    raw_cfg = proj.get(cfg_key)
    if isinstance(raw_cfg, str):
        cfg = json.loads(raw_cfg)
    elif isinstance(raw_cfg, dict):
        cfg = raw_cfg
    else:
        raise ValueError("Projection has %s weights but is missing '%s' metadata" % (prefix, cfg_key))

    sample_w = proj[seq_keys[0]]
    dtype = sample_w.dtype if device.type == "cuda" else torch.float32
    net = BidirectionalSeqConnector(
        d_in=cfg["d_in"],
        d_out=cfg["d_out"],
        d_model=cfg.get("d_model", 1536),
        n_layers=cfg.get("n_layers", 3),
        n_heads=cfg.get("n_heads", 12),
        ffn_mult=cfg.get("ffn_mult", 2),
        conv_kernel=cfg.get("conv_kernel", 5),
        dropout=0.0,
    )
    state = {k[len(prefix):]: proj[k] for k in seq_keys}
    net.load_state_dict(state, strict=True)
    net = net.to(device=device, dtype=dtype)
    net.train(False)
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def state_dict_prefixed(net, prefix="seq.", dtype=torch.bfloat16):
    """Export state_dict tensors with prefix for safetensors persistence."""
    return {
        prefix + k: v.detach().cpu().to(dtype=dtype).contiguous()
        for k, v in net.state_dict().items()
    }

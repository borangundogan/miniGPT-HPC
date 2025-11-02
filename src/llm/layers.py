from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from .rope import RoPECache, apply_rope_single
from .moe import MoE
from .rmsnorm import RMSNorm, make_norm


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int, activation: str = "gelu", dropout: float = 0.1, bias: bool = False):
        super().__init__()
        self.act_name = activation.lower()
        if self.act_name == "swiglu":
            self.w1 = nn.Linear(d_model, d_mlp, bias=bias)
            self.w2 = nn.Linear(d_model, d_mlp, bias=bias)
            self.proj = nn.Linear(d_mlp, d_model, bias=bias)
        else:
            self.fc = nn.Linear(d_model, d_mlp, bias=bias)
            self.proj = nn.Linear(d_mlp, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.act_name == "swiglu":
            a = self.w1(x)
            b = F.silu(self.w2(x))
            h = a * b
        else:
            h = self.fc(x)
            if self.act_name == "gelu":
                h = F.gelu(h)
            elif self.act_name == "silu":
                h = F.silu(h)
            else:
                raise ValueError(f"Unknown activation: {self.act_name}")
        h = self.proj(h)
        return self.dropout(h)


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        dropout: float = 0.1,
        bias: bool = False,
        use_rope: bool = True,
        max_seq_len: int = 2048,
        n_kv_head: Optional[int] = None,
    ):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        self.n_head = n_head
        self.n_kv_head = n_kv_head or n_head
        assert self.n_head % self.n_kv_head == 0, "n_head must be multiple of n_kv_head"

        self.d_head = d_model // n_head
        self.d_kv = self.d_head

        self.use_rope = use_rope
        self.max_seq_len = max_seq_len

        self.q_proj = nn.Linear(d_model, n_head * self.d_head, bias=bias)
        self.k_proj = nn.Linear(d_model, self.n_kv_head * self.d_kv, bias=bias)
        self.v_proj = nn.Linear(d_model, self.n_kv_head * self.d_kv, bias=bias)

        self.proj = nn.Linear(d_model, d_model, bias=bias)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # RoPE cache object
        self.rope_cache: Optional[RoPECache] = None

    def _maybe_init_rope(self, device):
        if self.use_rope and (self.rope_cache is None or self.rope_cache.device != device):
            self.rope_cache = RoPECache(self.d_head, self.max_seq_len, device=device)

    def forward(self, x, kv_cache=None, layer_idx: Optional[int] = None):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_head, self.d_head)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.d_kv)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.d_kv)

        Tprev = 0
        cached = None

        if kv_cache is not None and isinstance(kv_cache, tuple):
            k_prev, v_prev = kv_cache
            k_prev = k_prev.permute(0, 2, 1, 3)
            v_prev = v_prev.permute(0, 2, 1, 3)
            Tprev = k_prev.size(1)
            cached = (k_prev, v_prev)

        # Rotary embeddings
        if self.use_rope:
            self._maybe_init_rope(x.device)
            positions = torch.arange(Tprev, Tprev + T, device=x.device)
            cos, sin = self.rope_cache.get(positions)
            # apply_rope_single expects (B, H, T, D)
            q = apply_rope_single(q.transpose(1, 2), cos, sin).transpose(1, 2)
            k = apply_rope_single(k.transpose(1, 2), cos, sin).transpose(1, 2)

        if cached is not None:
            k_prev, v_prev = cached
            k = torch.cat([k_prev, k], dim=1)
            v = torch.cat([v_prev, v], dim=1)

        k_cache, v_cache = k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.n_kv_head < self.n_head:
            repeat_factor = self.n_head // self.n_kv_head
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)

        try:
            attn = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=True,
            )
        except Exception:
            Dh = q.size(-1)
            scores = (q @ k.transpose(-2, -1)) / (Dh ** 0.5)
            t_q, t_kv = scores.size(-2), scores.size(-1)
            i = torch.arange(t_q, device=x.device).unsqueeze(1)
            j = torch.arange(t_kv, device=x.device).unsqueeze(0)
            causal = j <= (Tprev + i)
            scores = scores.masked_fill(~causal, float("-inf"))
            attn_prob = torch.softmax(scores, dim=-1)
            if self.training and self.attn_drop.p > 0:
                attn_prob = F.dropout(attn_prob, p=self.attn_drop.p)
            attn = attn_prob @ v

        attn = attn.transpose(1, 2).contiguous().view(B, T, -1)
        out = self.proj(attn)
        out = self.resid_drop(out)
        return out, (k_cache, v_cache)


class HybridFFN(nn.Module):
    def __init__(
        self,
        dim: int,
        alpha: float = 0.5,
        mult: int = 4,
        swiglu: bool = True,
        n_expert: int = 4,
        k: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        inner = mult * dim
        self.dense = nn.Sequential(
            nn.Linear(dim, inner),
            nn.GELU(),
            nn.Linear(inner, dim),
            nn.Dropout(dropout),
        )
        self.moe = MoE(
            dim,
            n_expert=n_expert,
            k=k,
            mult=mult,
            swiglu=swiglu,
            dropout=dropout,
        )

    def forward(self, x):
        y_dense = self.dense(x)
        y_moe, aux = self.moe(x)
        y = self.alpha * y_dense + (1.0 - self.alpha) * y_moe
        return y, aux


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_mlp: int,
        dropout: float,
        bias: bool,
        norm_type: str,
        activation: str,
        use_rope: bool,
        max_seq_len: int,
        use_moe: bool = False,
        use_hybrid_ffn: bool = False,
        n_expert: int = 4,
        k_expert: int = 1,
        alpha: float = 0.5,
        n_kv_head: Optional[int] = None,
    ):
        super().__init__()
        self.ln1 = make_norm(d_model, norm_type)
        self.attn = CausalSelfAttention(
            d_model, n_head, dropout=dropout, bias=bias,
            use_rope=use_rope, max_seq_len=max_seq_len, n_kv_head=n_kv_head
        )
        self.ln2 = make_norm(d_model, norm_type)

        if use_hybrid_ffn:
            self.mlp = HybridFFN(
                dim=d_model,
                alpha=alpha,
                mult=d_mlp // d_model,
                n_expert=n_expert,
                k=k_expert,
                dropout=dropout,
            )
        elif use_moe:
            self.mlp = MoE(
                dim=d_model,
                n_expert=n_expert,
                k=k_expert,
                mult=d_mlp // d_model,
                swiglu=(activation.lower() == "swiglu"),
                dropout=dropout,
            )
        else:
            self.mlp = MLP(d_model, d_mlp, activation=activation, dropout=dropout, bias=bias)

        self.last_aux_loss = 0.0

    def forward(self, x, kv_cache=None, layer_idx: Optional[int] = None):
        attn_out, new_kv = self.attn(self.ln1(x), kv_cache=kv_cache, layer_idx=layer_idx)
        x = x + attn_out

        y = self.mlp(self.ln2(x))
        if isinstance(y, tuple):
            y, aux = y
            self.last_aux_loss = aux.mean() if torch.is_tensor(aux) else float(aux)
        else:
            self.last_aux_loss = 0.0
        x = x + y
        return x, new_kv

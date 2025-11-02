# src/llm/gpt.py
from typing import Optional
import torch
import torch.nn as nn
from .config import GPTConfig
from .layers import TransformerBlock, make_norm
from .kvcache import KVCache, LayerwiseKV


class GPTModel(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.use_rope:
            self.pos_emb = None
        else:
            self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=cfg.d_model,
                n_head=cfg.n_head,
                d_mlp=cfg.d_mlp,
                dropout=cfg.dropout,
                bias=cfg.bias,
                norm_type=cfg.norm_type,
                activation=cfg.activation,
                use_rope=cfg.use_rope,
                max_seq_len=cfg.max_seq_len,
                use_moe=cfg.use_moe,
                use_hybrid_ffn=cfg.use_hybrid_ffn,
                n_expert=cfg.n_expert,
                k_expert=cfg.k_expert,
                alpha=cfg.alpha,
            )
            for _ in range(cfg.n_layer)
        ])
        self.ln_f = make_norm(cfg.d_model, cfg.norm_type)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        self.lm_head.weight = self.tok_emb.weight
        self.lm_head.weight.requires_grad = True

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None, top_p=None):
        kv_cache = LayerwiseKV(
            self.cfg.n_layer,
            window=self.cfg.sliding_window or self.cfg.max_seq_len,
            sink=self.cfg.attention_sink,
        ) if self.cfg.use_kv_cache else None

        logits, new_kv_states = self(idx, kv_cache=kv_cache)
        logits = logits[:, -1, :]

        if kv_cache is not None:
            for i, (k, v) in enumerate(new_kv_states):
                kv_cache.append(i, k, v)

        for _ in range(max_new_tokens):
            logits, new_kv_states = self(idx[:, -1:], kv_cache=kv_cache)
            logits = logits[:, -1, :]

            if temperature != 1.0:
                logits /= temperature
            probs = torch.softmax(logits, dim=-1)

            if top_k:
                v, topk_idx = torch.topk(probs, min(top_k, probs.size(-1)))
                probs = probs.masked_fill(probs < v[:, [-1]], 0.0)
            if top_p:
                sorted_p, sort_idx = torch.sort(probs, descending=True)
                cum_p = torch.cumsum(sorted_p, dim=-1)
                mask = cum_p > top_p
                mask[:, 0] = False
                sorted_p = sorted_p.masked_fill(mask, 0.0)
                probs = torch.zeros_like(probs).scatter(-1, sort_idx, sorted_p)

            probs = probs / probs.sum(dim=-1, keepdim=True)
            next_id = torch.multinomial(probs, 1)

            idx = torch.cat([idx, next_id], dim=1)

            if kv_cache is not None:
                for i, (k, v) in enumerate(new_kv_states):
                    kv_cache.append(i, k, v)

        return idx

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ):
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}"

        x = self.tok_emb(idx)

        if self.pos_emb is not None:
            pos = torch.arange(0, T, device=idx.device)
            x = x + self.pos_emb(pos)[None, :, :]

        x = self.drop(x)
        new_kv_states = [] if kv_cache is not None else None

        for i, block in enumerate(self.blocks):
            layer_kv_cache = kv_cache.get(i) if kv_cache is not None else None
            x, new_layer_kv = block(x, kv_cache=layer_kv_cache, layer_idx=i)
            if kv_cache is not None:
                new_kv_states.append(new_layer_kv)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=-100,
            )
            return logits, loss

        return logits, new_kv_states

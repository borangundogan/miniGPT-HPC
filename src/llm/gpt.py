from typing import Optional
import torch
import torch.nn as nn
from .config import GPTConfig
from .layers import TransformerBlock, make_norm
from .kvcache import KVCache

class GPTModel(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        if cfg.use_rope:
            # RoPE replaces positional embeddings; keep a tiny learnable offset via bias-only PEs if desired later.
            self.pos_emb = None
        else:
            self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.d_model, cfg.n_head, cfg.d_mlp, cfg.dropout, cfg.bias, cfg.norm_type, cfg.activation, cfg.use_rope, cfg.max_seq_len)
            for _ in range(cfg.n_layer)
        ])
        self.ln_f = make_norm(cfg.d_model, cfg.norm_type)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=self.cfg.init_std)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                temperature: float = 1.0, top_k: Optional[int] = None,
                top_p: Optional[float] = None,
                eos_id: Optional[int] = None):
        self.eval()

        kv_cache = KVCache(
            self.cfg.n_layer,
            window=getattr(self.cfg, "sliding_window", None),
            sink=getattr(self.cfg, "attention_sink", 0),
        ) if self.cfg.use_kv_cache else None
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.max_seq_len:]
            logits = self(idx_cond, kv_cache=kv_cache)[:, -1, :]  # [B, vocab]

            # temperature
            if temperature != 1.0:
                logits = logits / max(1e-8, temperature)

            # top-k
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                thresh = v[:, [-1]]
                logits = torch.where(logits < thresh, torch.full_like(logits, float('-inf')), logits)

            # top-p
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = torch.softmax(sorted_logits, dim=-1)
                cumprobs = torch.cumsum(probs, dim=-1)
                # mask tokens beyond
                mask = cumprobs > top_p
                # keep first token even if > top_p
                mask[:, 0] = False
                sorted_logits = sorted_logits.masked_fill(mask, float('-inf'))
                # restore original order
                logits = torch.full_like(logits, float('-inf'))
                logits.scatter_(dim=-1, index=sorted_idx, src=sorted_logits)

            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)

            # early-stopping    
            if eos_id is not None:
                if (next_id == eos_id).all():
                    break
        
        return idx

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None, kv_cache: Optional[KVCache] = None):
        B, T = idx.shape
        assert T <= self.cfg.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.cfg.max_seq_len}"
        x = self.tok_emb(idx)  # [B,T,C]
        if self.pos_emb is not None:
            pos = torch.arange(0, T, device=idx.device)
            x = x + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)

        for i, block in enumerate(self.blocks):
            x = block(x, kv_cache=kv_cache, layer_idx=i if kv_cache is not None else None)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # [B,T,V]

        if targets is None:
            return logits
        # Causal LM loss
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

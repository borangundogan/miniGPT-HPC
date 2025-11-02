from __future__ import annotations
import torch, torch.nn as nn

class TopKGate(nn.Module):
    """Top-k softmax gating with Switch-style load-balancing aux loss."""
    def __init__(self, dim: int, n_expert: int, k: int = 1):
        super().__init__()
        assert 1 <= k <= n_expert
        self.n_expert = n_expert
        self.k = k
        self.w_g = nn.Linear(dim, n_expert, bias=True)

    def forward(self, x: torch.Tensor):
        # x: (S, C) where S = tokens
        logits = self.w_g(x)
        probs = torch.softmax(logits, dim=-1)               # (S,E)
        topk_vals, topk_idx = torch.topk(probs, self.k, dim=-1)

        # importance & load
        S, E = probs.shape
        importance = probs.mean(0)                           # (E,)
        hard1 = topk_idx[:, 0]
        load = torch.zeros(E, device=x.device, dtype=torch.float)
        load.scatter_add_(0, hard1, torch.ones(S, device=x.device))
        load = load / max(S, 1)
        aux_loss = E * torch.sum(importance * torch.clamp(load, min=1e-9))
        return topk_idx, topk_vals, aux_loss
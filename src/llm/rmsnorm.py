import torch
import torch.nn as nn
import torch.nn.functional as F

def make_norm(d_model: int, norm_type: str):
    if norm_type.lower() == "rmsnorm":
        return RMSNorm(d_model)
    elif norm_type.lower() == "layernorm":
        return nn.LayerNorm(d_model)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        norm_x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * norm_x

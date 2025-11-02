from __future__ import annotations
from typing import Optional
import torch


class KVCache:
    """Simple container for a single layer's KV tensors."""
    def __init__(self, k: torch.Tensor, v: torch.Tensor):
        self.k = k  # shape: (B, H, T, D)
        self.v = v  # shape: (B, H, T, D)

    @property
    def T(self):
        return self.k.size(2)


class RollingKV:
    """Rolling buffer for a single attention layer.
    Keeps first `sink` tokens and last `window` tokens.
    """
    def __init__(self, window: int, sink: int = 0):
        self.window = window
        self.sink = sink
        self.k = None
        self.v = None

    def step(self, k_new: torch.Tensor, v_new: torch.Tensor):
        """Append new keys/values and crop to window size if needed."""
        if self.k is None:
            self.k, self.v = k_new, v_new
        else:
            # append along time dimension (dim=2)
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)

        # crop if exceeds (window + sink)
        if self.k.size(2) > self.window + self.sink:
            sink_k = self.k[:, :, : self.sink, :] if self.sink > 0 else None
            sink_v = self.v[:, :, : self.sink, :] if self.sink > 0 else None
            tail_k = self.k[:, :, -self.window :, :]
            tail_v = self.v[:, :, -self.window :, :]

            if self.sink > 0:
                self.k = torch.cat([sink_k, tail_k], dim=2)
                self.v = torch.cat([sink_v, tail_v], dim=2)
            else:
                self.k, self.v = tail_k, tail_v

        return self.k, self.v


class LayerwiseKV:
    """Layer-wise KV cache wrapper for multi-layer GPT models."""
    def __init__(self, n_layer: int, window: int, sink: int = 0):
        self.layers = [RollingKV(window, sink) for _ in range(n_layer)]

    def get(self, layer_idx: int):
        kv = self.layers[layer_idx]
        if kv.k is None or kv.v is None:
            return None
        return kv.k, kv.v

    @torch.no_grad()
    def append(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor):
        """Append new K,V for a given layer."""
        return self.layers[layer_idx].step(k_new, v_new)

    def clear(self):
        """Reset all layers' caches."""
        for kv in self.layers:
            kv.k, kv.v = None, None

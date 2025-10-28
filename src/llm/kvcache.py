from typing import Optional, Tuple
import torch


class KVCache:
    """
    Layer-wise KV cache.
    - Default: append-only (growing) cache.
    - Optional rolling mode: keep first `sink` tokens + last `window` tokens (per layer),
      which bounds memory and compute to O(window + sink).
    """
    def __init__(self, n_layer: int, window: Optional[int] = None, sink: int = 0):
        self.keys = [None] * n_layer
        self.values = [None] * n_layer
        self.window = window          # if None => no rolling, grow indefinitely
        self.sink = int(sink) if sink is not None else 0

    def get(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        k, v = self.keys[layer_idx], self.values[layer_idx]
        if k is None or v is None:
            return None
        return k, v

    @torch.no_grad()
    def append(self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor):
        # k_new/v_new shape: [B, T_new, H, Dh]
        k_prev, v_prev = self.keys[layer_idx], self.values[layer_idx]
        if k_prev is None:
            k_cat, v_cat = k_new, v_new
        else:
            # concat along time dimension
            k_cat = torch.cat([k_prev, k_new], dim=1)
            v_cat = torch.cat([v_prev, v_new], dim=1)

        # rolling crop if enabled
        if self.window is not None:
            T = k_cat.size(1)
            keep = self.window + self.sink
            if T > keep:
                sink_k = k_cat[:, : self.sink] if self.sink > 0 else None
                sink_v = v_cat[:, : self.sink] if self.sink > 0 else None
                tail_k = k_cat[:, -self.window :] if self.window > 0 else None
                tail_v = v_cat[:, -self.window :] if self.window > 0 else None

                if self.sink > 0 and self.window > 0:
                    k_cat = torch.cat([sink_k, tail_k], dim=1)
                    v_cat = torch.cat([sink_v, tail_v], dim=1)
                elif self.sink > 0:
                    k_cat, v_cat = sink_k, sink_v
                else:
                    k_cat, v_cat = tail_k, tail_v

        self.keys[layer_idx] = k_cat
        self.values[layer_idx] = v_cat

    def clear(self):
        for i in range(len(self.keys)):
            self.keys[i], self.values[i] = None, None

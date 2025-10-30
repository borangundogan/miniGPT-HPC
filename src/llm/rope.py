import torch

def build_rope_cache(d_head: int, max_seq_len: int, device=None, base: float = 10000.0):
    #  ROPE (cos, sin) precomputed for speed
    theta = 1.0 / (base ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    pos = torch.arange(0, max_seq_len, device=device).float()
    idx = torch.outer(pos, theta)  # [T, d_head/2]
    cos = torch.cos(idx).repeat_interleave(2, dim=-1)  # [T, d_head]
    sin = torch.sin(idx).repeat_interleave(2, dim=-1)  # [T, d_head]
    return cos, sin

# TODO: didnt work correctly ! 
def apply_rope(x, cos, sin):
    # print("[apply_rope] x.shape=", tuple(x.shape), 
    #   "cos.shape=", tuple(cos.shape), "sin.shape=", tuple(sin.shape))

    # x: [B, T, n_head, d_head]
    T = x.size(1)

    if T == 0:
        return x
    
    if cos.size(0) < T:
        # pad the cos/sin if they're shorter than current sequence length
        pad = T - cos.size(0)
        cos = torch.cat([cos, cos[-1:].repeat(pad, 1)], dim=0)
        sin = torch.cat([sin, sin[-1:].repeat(pad, 1)], dim=0)

    x1, x2 = x[..., ::2], x[..., 1::2]
    x_rot = torch.stack((-x2, x1), dim=-1).reshape_as(x)
    cos_b = cos[None, :T, None, :]
    sin_b = sin[None, :T, None, :]
    return x * cos_b + x_rot * sin_b


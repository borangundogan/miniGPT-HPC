import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

@torch.no_grad()
def estimate_mfu(num_params, flops_per_token, tokens_per_second):
    # Rough and only illustrative MFU estimator; customize if needed.
    # MFU ~ (FLOPs actually used) / (peak FLOPs). Placeholder returns zeros.
    return 0.0

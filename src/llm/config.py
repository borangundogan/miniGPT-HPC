from dataclasses import dataclass
from typing import Optional

@dataclass
class GPTConfig:
    # Model dims
    vocab_size: int
    max_seq_len: int = 512
    n_layer: int = 6
    n_head: int = 8
    d_model: int = 512
    d_mlp: int = 2048

    # Regularization
    dropout: float = 0.1
    bias: bool = False

    # Positional encoding
    use_rope: bool = True   # if False, use learned positional embeddings

    # Norm / Act
    norm_type: str = "rmsnorm"   # "rmsnorm" | "layernorm"
    activation: str = "gelu"     # "gelu" | "silu" | "swiglu"

    # Inference
    use_kv_cache: bool = True

    # Init
    init_std: float = 0.02

    # Optional multi-query attention heads
    n_kv_head: Optional[int] = None

    # Sliding window attention
    sliding_window: Optional[int] = None
    attention_sink: int = 0

    use_moe: bool = False              # pure MoE mode
    use_hybrid_ffn: bool = False       # hybrid dense + MoE mode
    n_expert: int = 4                  # number of experts
    k_expert: int = 1                  # top-k routing
    alpha: float = 0.5                 # α blending for HybridFFN

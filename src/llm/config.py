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
    use_rope: bool = False   # if False, use learned positional embeddings

    # Norm / Act
    norm_type: str = "rmsnorm"   # "rmsnorm" | "layernorm"
    activation: str = "gelu"     # "gelu" | "silu" | "swiglu"

    # Inference
    use_kv_cache: bool = True

    # Init
    init_std: float = 0.02

    n_kv_head: Optional[int] = None 

    sliding_window: Optional[int] = None
    attention_sink: int = 0

    #MoE
    use_hybrid_ffn: bool = True
    hybrid_alpha: float = 0.5
    n_expert: int = 4
    k_expert: int = 1
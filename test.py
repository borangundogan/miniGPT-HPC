import torch
from src.llm import GPTConfig, GPTModel
from src.tokenizer.tokenizer_wrapper import TokenizerWrapper

# model path (trained checkpoint)
ckpt_path = "runs/test_hpc/model_best.pt"

# tokenizer
tokenizer = TokenizerWrapper("data/tokenizer/hpc_bpe.model")

# config
cfg = GPTConfig(
    vocab_size=tokenizer.vocab_size,
    max_seq_len=64,
    n_layer=6,
    n_head=8,
    d_model=512,
    d_mlp=2048,
    dropout=0.1,
    use_rope=False,
    activation="gelu",
    norm_type="rmsnorm",
    use_kv_cache=True,
)

# model + weights
model = GPTModel(cfg)
ckpt = torch.load(ckpt_path, map_location="cpu")
model.load_state_dict(ckpt["model"], strict=False)
model.eval()

# prompt
prompt = "In modern supercomputers, the OpenMP parallel region"
encoded = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)

# generation
with torch.no_grad():
    out = model.generate(encoded, max_new_tokens=60, temperature=0.8, top_p=0.9)
    text = tokenizer.decode(out[0].tolist())

print("\n=== MODEL OUTPUT ===")
print(text)

# test.py
import argparse, time, torch
from src.llm import GPTConfig, GPTModel
from src.tokenizer.tokenizer_wrapper import TokenizerWrapper

def main():
    ap = argparse.ArgumentParser(description="Generic LLM inference script.")
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--tok_model", type=str, required=True)
    ap.add_argument("--prompt", type=str, required=True)
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--use_kv_cache", type=int, default=1, help="1 = enable KV cache, 0 = disable")
    args = ap.parse_args()

    # Device
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")

    # Tokenizer
    tokenizer = TokenizerWrapper(args.tok_model)
    print(f"Loaded tokenizer | Vocab size: {tokenizer.vocab_size}")

    # Load checkpoint + model config
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg_dict = ckpt.get("config", {
        "vocab_size": tokenizer.vocab_size,
        "max_seq_len": 256,
        "n_layer": 6,
        "n_head": 8,
        "d_model": 512,
        "d_mlp": 2048,
        "dropout": 0.1,
    })
    cfg_dict["use_kv_cache"] = bool(args.use_kv_cache)
    cfg_dict.setdefault("n_kv_head", cfg_dict.get("n_head", 8))
    cfg_dict.setdefault("sliding_window", cfg_dict.get("max_seq_len", 256))
    cfg_dict.setdefault("attention_sink", 0)
    cfg = GPTConfig(**cfg_dict)
    model = GPTModel(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    print(f"Model loaded | Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    print(f"KV cache: {'ON' if cfg.use_kv_cache else 'OFF'}")

    # Encode + generate
    encoded = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long).to(device)

    torch.manual_seed(0)
    start = time.time()
    with torch.no_grad():
        out = model.generate(
            encoded,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
    end = time.time()

    text = tokenizer.decode(out[0].tolist())
    print("\n=== MODEL OUTPUT ===")
    print(text)
    print(f"\n[Generation time: {end - start:.2f}s | KV cache={'ON' if cfg.use_kv_cache else 'OFF'}]")

if __name__ == "__main__":
    main()

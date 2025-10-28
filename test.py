# test.py
import argparse, torch
from src.llm import GPTConfig, GPTModel
from src.tokenizer.tokenizer_wrapper import TokenizerWrapper

def main():
    ap = argparse.ArgumentParser(description="Generic LLM inference script.")
    ap.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.pt)")
    ap.add_argument("--tok_model", type=str, required=True, help="Path to tokenizer model (.model)")
    ap.add_argument("--prompt", type=str, required=True, help="Prompt text for generation")
    ap.add_argument("--max_new_tokens", type=int, default=80, help="Number of tokens to generate")
    ap.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    ap.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling (top-p)")
    ap.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    args = ap.parse_args()

    # Device selection
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
    cfg = GPTConfig(**cfg_dict)
    model = GPTModel(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    print(f"Model loaded | Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # Encode + generate
    encoded = torch.tensor([tokenizer.encode(args.prompt)], dtype=torch.long).to(device)
    with torch.no_grad():
        out = model.generate(
            encoded,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
    text = tokenizer.decode(out[0].tolist())

    print("\n=== MODEL OUTPUT ===")
    print(text)

if __name__ == "__main__":
    main()

# src/train/train.py
from __future__ import annotations
import argparse, os, time, math
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Optional

from llm import GPTConfig, GPTModel
from tokenizer.tokenizer_wrapper import TokenizerWrapper
from llm.corpus_dataset import CorpusDataset


@torch.no_grad()
def estimate_loss(model: GPTModel, dl_train, dl_val, device, eval_iters: int) -> dict:
    model.eval()
    out = {}
    def avg_loss(dloader):
        losses = []
        it = 0
        for x, y in dloader:
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
            it += 1
            if it >= eval_iters:
                break
        return sum(losses) / len(losses)
    out["train"] = avg_loss(dl_train)
    out["val"]   = avg_loss(dl_val)
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True, help="Path to text corpus file")
    ap.add_argument("--tok_model", type=str, default="data/tokenizer/hpc_bpe.model")
    ap.add_argument("--out_dir", type=str, default="runs/default_run", help="Directory under /runs for checkpoints")
    ap.add_argument("--max_seq_len", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--n_layer", type=int, default=6)
    ap.add_argument("--n_head", type=int, default=8)
    ap.add_argument("--d_model", type=int, default=512)
    ap.add_argument("--d_mlp", type=int, default=2048)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--eval_interval", type=int, default=200)
    ap.add_argument("--eval_iters", type=int, default=50)
    ap.add_argument("--sample_every", type=int, default=200)
    ap.add_argument("--sample_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--top_p", type=float, default=0.7)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    # Device setup
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else
        ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() and not args.cpu else "cpu")
    )
    print(f"Device: {device}")

    # Tokenizer
    tokenizer = TokenizerWrapper(args.tok_model)

    # Datasets
    ds_train = CorpusDataset(args.data, tokenizer, args.max_seq_len, split="train")
    ds_val   = CorpusDataset(args.data, tokenizer, args.max_seq_len, split="val")
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, drop_last=True)
    dl_val   = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False, drop_last=True)

    # Model config
    cfg = GPTConfig(
        vocab_size=tokenizer.vocab_size,
        max_seq_len=args.max_seq_len,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_model=args.d_model,
        d_mlp=args.d_mlp,
        dropout=args.dropout,
        use_rope=True,
        activation="gelu",
        norm_type="rmsnorm",
        use_kv_cache=True,
    )
    model = GPTModel(cfg).to(device)
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)
    print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    

    # Optimizer + AMP
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device.type == "cuda"))

    # Output directory under root /runs/
    out_dir = os.path.join("runs", os.path.basename(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    print(f"Checkpoints will be saved to: {out_dir}")

    # Training loop
    best_val = float("inf")
    t0 = time.time()
    model.train()
    step = 0
    train_iter = iter(dl_train)

    while step < args.steps:
        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(dl_train)
            continue

        x, y = x.to(device), y.to(device)
        with torch.cuda.amp.autocast(enabled=(args.amp and device.type == "cuda")):
            _, loss = model(x, y)
            
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()

        step += 1
        if step % 10 == 0:
            print(f"step {step:5d} | loss {loss.item():.4f} | {time.time()-t0:.1f}s")
            t0 = time.time()

        # Eval + checkpoint
        if step % args.eval_interval == 0:
            scores = estimate_loss(model, dl_train, dl_val, device, args.eval_iters)
            print(f"eval | train {scores['train']:.4f} | val {scores['val']:.4f}")
            if scores["val"] < best_val:
                best_val = scores["val"]
                ckpt_path = os.path.join(out_dir, "model_best.pt")
                torch.save({
                    "model": model.state_dict(),
                    "config": {
                        "vocab_size": tokenizer.vocab_size,
                        "max_seq_len": args.max_seq_len,
                        "n_layer": args.n_layer,
                        "n_head": args.n_head,
                        "d_model": args.d_model,
                        "d_mlp": args.d_mlp,
                        "dropout": args.dropout,
                    }
                }, ckpt_path)
                print(f"saved checkpoint: {ckpt_path}")

        # Sample
        if args.sample_every > 0 and step % args.sample_every == 0:
            model.eval()
            with torch.no_grad():
                max_start = max(1, len(ds_train.ids) - args.max_seq_len - 1)
                start = torch.randint(low=0, high=max_start, size=(1,)).item()
                seed = ds_train.ids[start:start + args.max_seq_len].unsqueeze(0).to(device)
                out = model.generate(seed,
                                     max_new_tokens=args.sample_tokens,
                                     temperature=args.temperature,
                                     top_k=args.top_k,
                                     top_p=args.top_p)
                txt = tokenizer.decode(out[0].tolist())
                print("\n================ SAMPLE ================\n" +
                      txt[-(args.max_seq_len + args.sample_tokens):] +
                      "\n=======================================\n")
            model.train()

    # Final save
    final_path = os.path.join(out_dir, "model_final.pt")
    torch.save({"model": model.state_dict()}, final_path)
    print(f"✅ Saved final model: {final_path}")

if __name__ == "__main__":
    main()

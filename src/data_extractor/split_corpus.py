import argparse
from pathlib import Path
import random

def split_corpus(corpus_path: Path, train_ratio=0.95, val_ratio=0.025, seed=42):
    """
    Split corpus into train/val/test by <|endoftext|>-separated documents.
    Keeps documents intact to preserve context boundaries.
    """
    if not corpus_path.exists():
        raise FileNotFoundError(f"File not found: {corpus_path}")

    # Read and segment corpus by document delimiter
    raw = corpus_path.read_text(encoding="utf-8")
    docs = [d.strip() for d in raw.split("<|endoftext|>") if len(d.strip()) > 50]
    if not docs:
        raise ValueError("No valid documents found in corpus.")

    random.seed(seed)
    random.shuffle(docs)

    n_total = len(docs)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    train_docs = docs[:n_train]
    val_docs = docs[n_train:n_train + n_val]
    test_docs = docs[n_train + n_val:]

    def save_split(name: str, subset: list[str]):
        out_path = corpus_path.parent / f"{corpus_path.stem}_{name}.txt"
        text = "\n\n<|endoftext|>\n\n".join(subset) + "\n\n<|endoftext|>\n"
        out_path.write_text(text, encoding="utf-8")
        print(f"Saved {name} split ({len(subset)} documents) to {out_path}")

    save_split("train", train_docs)
    save_split("val", val_docs)
    save_split("test", test_docs)

    print(f"Total: {n_total} documents | Train: {n_train} | Val: {n_val} | Test: {n_test}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a shuffled corpus into train/val/test sets.")
    parser.add_argument("--input", type=str, required=True, help="Path to shuffled corpus file")
    parser.add_argument("--train_ratio", type=float, default=0.95, help="Proportion of data for training")
    parser.add_argument("--val_ratio", type=float, default=0.025, help="Proportion of data for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    split_corpus(Path(args.input), args.train_ratio, args.val_ratio, args.seed)

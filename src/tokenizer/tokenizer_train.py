# src/tokenizer/tokenizer_train.py
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer


def train_hf_bpe(
    corpus_path: Path,
    out_dir: Path,
    vocab_size: int = 8000,
    min_frequency: int = 2,
):
    """
    Train a Hugging Face Byte-Level BPE tokenizer on the given corpus.
    """
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus not found: {corpus_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Training HuggingFace Byte-Level BPE Tokenizer ===")
    print(f"Corpus: {corpus_path}")
    print(f"Output: {out_dir}")
    print(f"Vocab size: {vocab_size}")

    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train(
        files=[str(corpus_path)],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<s>", "</s>", "<pad>", "<unk>", "<mask>"]
    )

    tokenizer.save_model(str(out_dir))
    tokenizer.save(str(out_dir / "tokenizer.json"))

    print(f"Tokenizer saved to {out_dir}/vocab.json, merges.txt, tokenizer.json")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=8000)
    parser.add_argument("--min_frequency", type=int, default=2)

    args = parser.parse_args()

    train_hf_bpe(
        corpus_path=Path(args.input),
        out_dir=Path(args.output_dir),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )
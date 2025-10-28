import argparse
import random
from pathlib import Path

def shuffle_corpus(corpus_path: Path):
    """Shuffle <|endoftext|>-separated documents while preserving internal order."""
    if not corpus_path.exists():
        raise FileNotFoundError(f"❌ Corpus file not found: {corpus_path}")

    shuffled_path = corpus_path.with_name(corpus_path.stem + "_shuffled.txt")

    # Read and split into document blocks
    raw = corpus_path.read_text(encoding="utf-8")
    docs = [d.strip() for d in raw.split("<|endoftext|>") if len(d.strip()) > 50]

    if not docs:
        raise ValueError("❌ No valid documents found in corpus.")

    random.shuffle(docs)

    # Reassemble with delimiters
    shuffled_text = "\n\n<|endoftext|>\n\n".join(docs) + "\n\n<|endoftext|>\n"
    shuffled_path.write_text(shuffled_text, encoding="utf-8")

    print(f"✅ Shuffled {len(docs)} documents.")
    print(f"💾 Saved to: {shuffled_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shuffle a corpus file by <|endoftext|> document blocks.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to the corpus .txt file (e.g., data/corpus/hpc_corpus_20251101_154522.txt)",
    )
    args = parser.parse_args()

    shuffle_corpus(Path(args.input))

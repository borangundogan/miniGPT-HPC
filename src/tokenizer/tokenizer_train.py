import argparse
import sentencepiece as spm
from pathlib import Path


def train_tokenizer(
    corpus_path: Path,
    model_prefix: str,
    vocab_size: int = 8000,
    character_coverage: float = 1.0,
    max_sentence_length: int = 40000,
    input_sentence_size: int = 500000,
    shuffle: bool = True,
):
    """
    Train a SentencePiece tokenizer (BPE) on a given text corpus.
    """
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    print(f"Starting tokenizer training...")
    print(f"Input corpus: {corpus_path}")
    print(f"Output model prefix: {model_prefix}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Character coverage: {character_coverage}")
    print(f"Max sentence length: {max_sentence_length}")
    print(f"Input sentence size: {input_sentence_size}")
    print(f"Shuffle input: {shuffle}")

    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=character_coverage,
        max_sentence_length=max_sentence_length,
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=shuffle,
        byte_fallback=True,         # ensures stable behavior for unseen symbols
        hard_vocab_limit=False,     # allows flexible vocab adjustment
    )

    print(f"Tokenizer training completed. Model saved to {model_prefix}.model / .vocab")


def quick_test(model_prefix: Path, text: str = None):
    """
    Quick sanity test to verify tokenizer behavior.
    """
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(str(model_prefix) + ".model")

    if text is None:
        text = "The OpenMP parallel region uses shared memory and SIMD instructions."

    print("\n--- Tokenizer Sanity Test ---")
    print(f"Input text: {text}")
    print("Encoded tokens:", sp.encode(text, out_type=str))
    print("Decoded text:", sp.decode(sp.encode(text, out_type=int)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a SentencePiece tokenizer on a given corpus.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input corpus (.txt)")
    parser.add_argument("--out_dir", type=str, default="data/tokenizer", help="Output directory for model files")
    parser.add_argument("--prefix", type=str, default="hpc_bpe", help="Prefix for model and vocab files")
    parser.add_argument("--vocab_size", type=int, default=8000, help="Vocabulary size for SentencePiece")
    parser.add_argument("--char_coverage", type=float, default=1.0, help="Fraction of characters covered by model")
    parser.add_argument("--max_sent_len", type=int, default=40000, help="Max sentence length for training")
    parser.add_argument("--input_sent_size", type=int, default=500000, help="Number of sampled sentences for training")
    parser.add_argument("--no_shuffle", action="store_true", help="Disable shuffling during training")
    parser.add_argument("--test_text", type=str, default=None, help="Optional text to run a post-train test")

    args = parser.parse_args()

    corpus_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_prefix = out_dir / args.prefix

    train_tokenizer(
        corpus_path=corpus_path,
        model_prefix=model_prefix,
        vocab_size=args.vocab_size,
        character_coverage=args.char_coverage,
        max_sentence_length=args.max_sent_len,
        input_sentence_size=args.input_sent_size,
        shuffle=not args.no_shuffle,
    )

    quick_test(model_prefix, args.test_text)

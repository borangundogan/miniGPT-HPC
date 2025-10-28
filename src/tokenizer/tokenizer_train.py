import sentencepiece as spm
from pathlib import Path

# CONFIGURATION
CORPUS_DIR = Path("data/corpus")
TOKENIZER_DIR = Path("data/tokenizer")
TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PREFIX = TOKENIZER_DIR / "hpc_bpe"
VOCAB_SIZE = 8000        # ideal for a small domain corpus and local training
CHAR_COVERAGE = 1.0      # HPC corpus is mostly ASCII
MAX_SENT_LEN = 40000     # prevents skipping long lines during training # 20000 


def get_latest_corpus(corpus_dir: Path = CORPUS_DIR) -> Path:
    """Return the most recently created corpus file (hpc_corpus_*.txt)."""
    corpus_files = sorted(
        corpus_dir.glob("hpc_corpus_*.txt"),
        key=lambda f: f.stat().st_mtime,
        reverse=True
    )
    if not corpus_files:
        raise FileNotFoundError(f"No corpus files found in {corpus_dir}")
    latest = corpus_files[0]
    print(f"Using latest corpus: {latest.name}")
    return latest


def train_tokenizer():
    """Train a BPE tokenizer on the latest available corpus."""
    corpus_path = get_latest_corpus()
    print(f"Starting tokenizer training on corpus: {corpus_path}")
    print(f"Target vocabulary size: {VOCAB_SIZE}")

    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=str(MODEL_PREFIX),
        vocab_size=VOCAB_SIZE,
        model_type="bpe",
        character_coverage=CHAR_COVERAGE,
        input_sentence_size=500000,
        shuffle_input_sentence=True,
        max_sentence_length=MAX_SENT_LEN,
        # Optional: enable byte fallback for rare symbols
        # byte_fallback=True,
        # Optional: allow flexible vocabulary limit
        # hard_vocab_limit=False,
    )

    print(f"Tokenizer training completed. Files saved to {TOKENIZER_DIR}")


def quick_test():
    """Run a simple tokenization test using a sample HPC-related sentence."""
    sp = spm.SentencePieceProcessor()
    sp.load(str(MODEL_PREFIX) + ".model")

    sample_text = "The OpenMP parallel region uses shared memory and SIMD instructions."
    tokens = sp.encode(sample_text, out_type=str)

    print("\nSample text:")
    print(sample_text)
    print("\nTokenized output:")
    print(tokens)


if __name__ == "__main__":
    train_tokenizer()
    quick_test()

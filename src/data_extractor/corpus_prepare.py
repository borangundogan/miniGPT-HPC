import re
from pathlib import Path
from datetime import datetime

# CONFIGURATION
TEXT_DIR = Path("data/text")
CORPUS_DIR = Path("data/corpus")
CORPUS_DIR.mkdir(parents=True, exist_ok=True)

# timestamped output
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
FINAL_CORPUS_PATH = CORPUS_DIR / f"hpc_corpus_{timestamp}.txt"

MAX_LINE_LENGTH = 2000   # SentencePiece 
MIN_LINE_LENGTH = 50     # pass too short sentences


def merge_and_clean_texts(text_dir: Path = TEXT_DIR) -> str:
    """Merge and clean all text files into one string."""
    all_texts = []
    for file in sorted(text_dir.glob("*.txt")):
        text = open(file, encoding="utf-8").read().strip()
        if len(text) < MIN_LINE_LENGTH:
            continue
        # normalize whitespace and remove non-ascii
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\x00-\x7F]+", " ", text)
        all_texts.append(text + "\n\n<|endoftext|>\n\n")
    merged = "\n".join(all_texts)
    print(f"Merged {len(all_texts)} text files.")
    return merged


def split_long_sentences(text: str) -> str:
    """Split overly long lines to keep SentencePiece training stable."""
    sentences = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        sentence = sentence.strip()
        if not sentence or len(sentence) < MIN_LINE_LENGTH:
            continue
        if len(sentence) > MAX_LINE_LENGTH:
            # further split on commas or semicolons
            parts = re.split(r"[,:;]", sentence)
            for part in parts:
                part = part.strip()
                if MIN_LINE_LENGTH < len(part) <= MAX_LINE_LENGTH:
                    sentences.append(part)
        else:
            sentences.append(sentence)
    print(f"Cleaned and split into {len(sentences)} lines.")
    return "\n".join(sentences)


def prepare_corpus():
    """Full pipeline: merge, clean, split, and save timestamped corpus."""
    if not list(TEXT_DIR.glob("*.txt")):
        print(f"No .txt files found in {TEXT_DIR}")
        return

    merged_text = merge_and_clean_texts(TEXT_DIR)
    final_text = split_long_sentences(merged_text)
    FINAL_CORPUS_PATH.write_text(final_text, encoding="utf-8")

    size_mb = len(final_text) / 1e6
    print(f"Saved final corpus: {FINAL_CORPUS_PATH} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    prepare_corpus()

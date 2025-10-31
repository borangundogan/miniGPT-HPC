import re
import argparse
from pathlib import Path
from datetime import datetime
import unicodedata

def fix_hyphenation(text: str) -> str:
    # "pre - cision" or "mod - elling" → "precision"
    text = re.sub(r'(\w)\s*-\s*(\w)', r'\1\2', text)

    text = re.sub(r'\s*-\s*', '-', text)
    
    return text


def merge_and_clean_texts(text_dir: Path, min_len: int, max_len: int) -> str:
    """Merge and clean all text files in a directory into one corpus string."""
    all_texts = []

    txt_files = list(text_dir.glob("*.txt"))
    if not txt_files:
        print(f"No .txt files found in {text_dir}")
        return ""

    for file in sorted(txt_files):
        with open(file, encoding="utf-8") as f:
            text = f.read().strip()

        if len(text) < min_len:
            continue

        text = fix_hyphenation(text)

        # Normalize whitespace and Unicode
        text = re.sub(r"\s+", " ", text)
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
        all_texts.append(text + "\n\n<|endoftext|>\n\n")

    merged = "\n".join(all_texts)
    print(f"Merged {len(all_texts)} files from {text_dir.name}.")
    return merged


def split_long_sentences(text: str, min_len: int, max_len: int) -> str:
    """Split overly long sentences to stabilize tokenizer training."""
    sentences = []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        sentence = sentence.strip()
        if not sentence or len(sentence) < min_len:
            continue
        if len(sentence) > max_len:
            parts = re.split(r"[,:;]", sentence)
            for part in parts:
                part = part.strip()
                if min_len < len(part) <= max_len:
                    sentences.append(part)
        else:
            sentences.append(sentence)

    print(f"🧹 Cleaned and split into {len(sentences)} lines.")
    return "\n".join(sentences)


def prepare_corpus(input_path: Path, output_dir: Path, min_len: int, max_len: int):
    """Full pipeline: merge, clean, split, and save timestamped corpus."""
    if input_path.is_file() and input_path.suffix == ".txt":
        # Only process the specified file
        with open(input_path, encoding="utf-8") as f:
            text = f.read().strip()
        print(f"✅ Loaded single file: {input_path.name}")

        text = re.sub(r"\s+", " ", text)
        text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
        merged_text = text + "\n\n<|endoftext|>\n\n"

    else:
        # Process all .txt files under the folder
        text_dir = input_path
        merged_text = merge_and_clean_texts(text_dir, min_len, max_len)

    final_text = split_long_sentences(merged_text, min_len, max_len)

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"hpc_corpus_{timestamp}.txt"

    out_path.write_text(final_text, encoding="utf-8")

    size_mb = len(final_text) / 1e6
    num_segments = final_text.count("<|endoftext|>")
    print(f"💾 Saved corpus: {out_path} | {size_mb:.2f} MB | {num_segments} segments")



if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Merge and clean raw HPC text files into one corpus.")
    ap.add_argument("--input", type=str, required=True,
                    help="Input path (a single .txt file or a directory containing .txt files)")
    ap.add_argument("--output", type=str, default="data/corpus",
                    help="Output directory for the final corpus file")
    ap.add_argument("--min_len", type=int, default=50, help="Minimum sentence length to keep")
    ap.add_argument("--max_len", type=int, default=2000, help="Maximum sentence length before splitting")

    args = ap.parse_args()
    prepare_corpus(Path(args.input), Path(args.output), args.min_len, args.max_len)

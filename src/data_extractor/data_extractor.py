import os
import re
from pathlib import Path
from PyPDF2 import PdfReader

# === PATH CONFIGURATION ===
PDF_DIR = Path("data/pdfs")
TEXT_DIR = Path("data/text")
TEXT_DIR.mkdir(parents=True, exist_ok=True)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract raw text from a single PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text() or ""
        text += page_text + "\n"
    return text


def clean_text(text: str) -> str:
    """Apply generic cleaning suitable for HPC lecture or technical PDFs."""
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    # Remove slide or course metadata
    text = re.sub(r"PTfS\s*\d{4}", " ", text)
    text = re.sub(r"May\s+\d{1,2},\s*\d{4}", " ", text)
    text = re.sub(r"Programming Techniques for Supercomputers", " ", text)

    # Replace unwanted symbols
    text = (
        text.replace("", "->")
        .replace("→", "->")
        .replace("–", "-")
        .replace("", "")
        .replace("", "-")
        .replace("•", "-")
    )

    # Add spacing between joined words (e.g., SingleInstruction -> Single Instruction)
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

    # Collapse multiple spaces
    text = re.sub(r"\s{2,}", " ", text).strip()

    return text


def process_all_pdfs(pdf_dir: Path = PDF_DIR, out_dir: Path = TEXT_DIR):
    """Extract and clean text from all PDFs in the directory."""
    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {pdf_dir}")
        return

    print(f"Found {len(pdf_files)} PDF files in {pdf_dir}")

    for pdf_file in pdf_files:
        print(f"Processing: {pdf_file.name}")
        try:
            raw_text = extract_text_from_pdf(pdf_file)
            cleaned = clean_text(raw_text)
            out_path = out_dir / f"{pdf_file.stem}.txt"
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(cleaned)
            print(f"Saved cleaned text to {out_path}")
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")


if __name__ == "__main__":
    process_all_pdfs()

import random, re
import torch
from torch.utils.data import Dataset
from tokenizer.tokenizer_wrapper import TokenizerWrapper


class CorpusDataset(Dataset):
    """
    A dataset that reads a text corpus (e.g., HPC lecture notes),
    tokenizes it, and produces (x, y) token pairs for next-token prediction.
    """
    def __init__(self, corpus_path: str, tokenizer: TokenizerWrapper,
                 max_seq_len: int, split: str = "train", split_ratio: float = 0.9):

        with open(corpus_path, "r", encoding="utf-8") as f:
            paragraphs = re.split(r"\n\s*\n", f.read().strip())

        n = int(len(paragraphs) * split_ratio)
        selected_paras = paragraphs[:n] if split == "train" else paragraphs[n:]

        if len(selected_paras) < 5:
            raise ValueError("Validation set too small; adjust split_ratio or corpus size.")

        # text = " ".join(selected_paras)
        text = "\n\n".join(selected_paras)  # or "\n" if paragraphs already double-spaced
        self.ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
        self.max_seq_len = max_seq_len

        print(f"[CorpusDataset] Split={split} | Tokens={len(self.ids):,} | Paragraphs={len(selected_paras)}")

    def __len__(self):
        return max(0, len(self.ids) - self.max_seq_len)

    def __getitem__(self, idx):
        x = self.ids[idx : idx + self.max_seq_len]
        y = self.ids[idx + 1 : idx + 1 + self.max_seq_len]
        return x, y

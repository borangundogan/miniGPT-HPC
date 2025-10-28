import torch
from torch.utils.data import Dataset
from tokenizer.tokenizer_wrapper import TokenizerWrapper

class CorpusDataset(Dataset):
    """
    A simple dataset class that reads a single large text file
    (e.g., HPC lecture notes), tokenizes it, and generates (x, y)
    pairs using a sliding window approach.

    Args:
        corpus_path (str): Path to the text corpus file (.txt).
        tokenizer (TokenizerWrapper): The tokenizer instance used
            to convert text into token IDs.
        max_seq_len (int): Maximum sequence length (window size).
        split (str): Which split to load, "train" or "val".
        split_ratio (float): Fraction of data used for training.
            The remainder is used for validation. Default = 0.9.
    """
    def __init__(self, corpus_path: str, tokenizer: TokenizerWrapper,
                 max_seq_len: int, split: str = "train", split_ratio: float = 0.9):
        with open(corpus_path, "r", encoding="utf-8") as f:
            text = f.read()

        ids = tokenizer.encode(text)
        n = int(len(ids) * split_ratio)
        if split == "train":
            self.ids = torch.tensor(ids[:n], dtype=torch.long)
        else:
            self.ids = torch.tensor(ids[n:], dtype=torch.long)
        self.max_seq_len = max_seq_len

    def __len__(self):
        return max(1, len(self.ids) - self.max_seq_len - 1)

    def __getitem__(self, idx):
        x = self.ids[idx : idx + self.max_seq_len]
        y = self.ids[idx + 1 : idx + 1 + self.max_seq_len]
        return x, y

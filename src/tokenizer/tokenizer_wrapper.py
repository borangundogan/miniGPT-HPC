# src/tokenizer/tokenizer_wrapper.py
from pathlib import Path
import torch
from tokenizers import Tokenizer


class TokenizerWrapper:
    """
    Wrapper for Hugging Face Byte-Level BPE tokenizer.
    Supports encode/decode and batch tensorization.
    """

    def __init__(self, tokenizer_dir: str):
        tokenizer_path = Path(tokenizer_dir)
        if tokenizer_path.is_dir():
            tokenizer_path = tokenizer_path / "tokenizer.json"
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")

        self.tok = Tokenizer.from_file(str(tokenizer_path))
        self.vocab_size = self.tok.get_vocab_size()

        self.pad_id = self.tok.token_to_id("<pad>") or 0
        self.bos_id = self.tok.token_to_id("<s>") or 0
        self.eos_id = self.tok.token_to_id("</s>") or 0
        self.unk_id = self.tok.token_to_id("<unk>") or 0

        if None in [self.pad_id, self.bos_id, self.eos_id, self.unk_id]:
            raise ValueError("Didnt find special tokens ID!")

        print(f"Loaded tokenizer from {tokenizer_path}")
        print(f"Vocab size: {self.vocab_size}")

    def encode(self, text: str, add_special_tokens: bool = True):
        ids = self.tok.encode(text).ids
        if add_special_tokens:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids):
        return self.tok.decode(ids)

    def encode_batch(self, texts, max_length=512):
        encoded = [self.encode(t)[:max_length] for t in texts]
        max_len = max(len(x) for x in encoded)

        input_ids, attention_mask = [], []
        for seq in encoded:
            pad_len = max_len - len(seq)
            input_ids.append(seq + [self.pad_id] * pad_len)
            attention_mask.append([1] * len(seq) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def decode_batch(self, batch_ids):
        return [self.decode(ids) for ids in batch_ids]

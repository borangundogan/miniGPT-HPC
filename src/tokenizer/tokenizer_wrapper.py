from pathlib import Path
import torch
import sentencepiece as spm

class TokenizerWrapper:
    """Utility wrapper around SentencePiece for easy encode/decode and batching."""

    def __init__(self, model_path: str):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)

        # Special tokens
        self.pad_id = 0
        self.bos_id = 1
        self.eos_id = 2
        self.unk_id = 3 if self.sp.unk_id() == -1 else self.sp.unk_id()

        print(f"Loaded tokenizer from {model_path}")
        print(f"Vocab size: {self.vocab_size}")

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """Encode a single text string into token IDs."""
        ids = self.sp.encode(text, out_type=int)
        if add_special_tokens:
            return [self.bos_id] + ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back into text."""
        return self.sp.decode(ids)

    def encode_batch(self, texts: list[str], max_length: int = 512) -> dict[str, torch.Tensor]:
        """Encode and pad a batch of texts into tensors."""
        encoded = [self.encode(t)[:max_length] for t in texts]
        max_len = max(len(x) for x in encoded)

        input_ids = []
        attention_mask = []

        for seq in encoded:
            pad_len = max_len - len(seq)
            input_ids.append(seq + [self.pad_id] * pad_len)
            attention_mask.append([1] * len(seq) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }

    def decode_batch(self, batch_ids: list[list[int]]) -> list[str]:
        """Decode a batch of token ID sequences."""
        return [self.decode(ids) for ids in batch_ids]
    
    @property
    def vocab_size(self) -> int:
        """Return total vocabulary size from SentencePiece model."""
        return self.sp.get_piece_size()
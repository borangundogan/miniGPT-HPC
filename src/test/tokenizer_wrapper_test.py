from tokenizer.tokenizer_wrapper import TokenizerWrapper

tokenizer = TokenizerWrapper("data/tokenizer/hpc_bpe.model")

# Single text
encoded = tokenizer.encode("The OpenMP parallel region uses shared memory.")
print(encoded)
print(tokenizer.decode(encoded))

# Batch encode
batch = tokenizer.encode_batch([
    "OpenMP uses shared memory parallelization.",
    "Cache blocking improves throughput."
])
print(batch["input_ids"].shape)      # torch.Size([2, seq_len])
print(batch["attention_mask"].shape)

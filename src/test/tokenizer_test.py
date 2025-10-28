import sentencepiece as spm

sp = spm.SentencePieceProcessor()
sp.load("data/tokenizer/hpc_bpe_long_sent.model")

print(sp.encode("OpenMP uses shared memory parallelization.", out_type=str))
print(sp.encode("MPI_Allreduce synchronizes data across nodes.", out_type=str))
print(sp.encode("Cache blocking improves locality and throughput.", out_type=str))

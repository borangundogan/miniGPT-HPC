# HPC-LM — Domain-Specific GPT for High-Performance Computing

HPC-LM v2 is an end-to-end framework for training compact GPT-style language models specialized in High-Performance Computing (HPC) texts such as OpenMP, MPI, CUDA, and NUMA documentation.
It now supports Rotary Positional Embeddings, Mixture-of-Experts (MoE) layers, Hybrid Feed-Forward Networks, and KV-Cache for fast inference.

## Highlights
- Transform HPC lecture notes, manuals, and PDFs into a clean training corpus.
- Train a Byte-Level BPE tokenizer tuned for HPC jargon.
- Build and train a GPT architecture with:
  - Rotary Positional Embeddings (RoPE)
  - Mixture-of-Experts (MoE) & Hybrid FFN
  - KV-Cache for efficient generation
  - Mixed Precision (AMP) and `torch.compile` acceleration
- Evaluate & sample during training with real-time text outputs.

## Prerequisites
- Python 3.10+
- PyTorch 2.1+ (CUDA / MPS / CPU supported)
- `uv` for environment management → [astral-sh/uv](https://github.com/astral-sh/uv)
- Prepared corpus under `data/corpus/`

## Quickstart
1. **Install dependencies**
   ```bash
   uv sync
   ```

2. **Prepare corpus**
   ```bash
   PYTHONPATH=src uv run python src/data_extractor/data_extractor.py
   PYTHONPATH=src uv run python src/data_extractor/corpus_prepare.py
   uv run python src/data_extractor/shuffle_corpus.py --input data/corpus/hpc_corpus_YYYYMMDD_HHMMSS.txt
   ```

3. **Train tokenizer (Hugging Face BPE)**
   ```bash
   uv run python src/tokenizer/tokenizer_train.py \
     --input data/corpus/hpc_corpus_clean.txt \
     --output_dir data/tokenizer \
     --vocab_size 8000
   ```
   Artifacts: `tokenizer.json`, `vocab.json`, `merges.txt`

4. **Launch training**

   Example: full-feature MoE + RoPE + KV Cache + AMP
   ```bash
   PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 uv run python src/train/train.py \
     --data data/corpus/hpc_corpus_clean.txt \
     --tok_model data/tokenizer/ \
     --out_dir runs/hpc_moe_rope_v2 \
     --max_seq_len 256 \
     --batch_size 16 \
     --steps 8000 \
     --dropout 0.1 \
     --lr 3e-4 \
     --eval_interval 500 \
     --sample_every 500 \
     --use_rope \
     --use_kv_cache \
     --use_moe \
     --use_hybrid_ffn \
     --n_expert 6 \
     --k_expert 2 \
     --alpha 0.5 \
     --lambda_aux 5e-3 \
     --amp \
     --compile
   ```

## Model features

| Feature | Flag | Status |
| --- | --- | --- |
| Rotary Positional Embedding | `--use_rope` | ✅ |
| Mixture of Experts | `--use_moe` | ✅ |
| Hybrid FFN (Dense + MoE) | `--use_hybrid_ffn` | ✅ |
| KV-Cache (faster generation) | `--use_kv_cache` | ✅ |
| AMP / FP16 | `--amp` | ✅ |
| Torch Compile | `--compile` | ✅ |

5. **Evaluate and generate samples**

During training, samples are periodically printed:
```
================ SAMPLE ================
In modern supercomputers, the OpenMP parallel region
enables threads to share memory and synchronize barriers
...
=======================================
```

6. **Inference (text generation)**
```bash
PYTHONPATH=src CUDA_VISIBLE_DEVICES=0 python src/test/test.py \
  --ckpt runs/hpc_moe_rope_v2/model_best.pt \
  --tok_model data/tokenizer/ \
  --prompt "In modern supercomputers, the OpenMP parallel region" \
  --max_new_tokens 120 \
  --temperature 0.8 \
  --top_p 0.95 \
  --top_k 50 \
  --use_kv_cache 1
```

Example output:
```
=== MODEL OUTPUT ===
In modern supercomputers, the OpenMP parallel region allows
threads to execute concurrently across shared memory systems...
[Generation time: 1.9 s | KV cache = ON]
```

## Troubleshooting
| Issue | Fix |
| --- | --- |
| Size mismatch when resuming | Change `--out_dir` or match `--n_expert`, `--k_expert` to checkpoint. |
| Aux loss not decreasing | Ensure `last_aux_loss` stays a `Tensor` (no `.item()` in `MoE.forward()`). |
| AMP deprecation warnings | Replace `torch.cuda.amp.autocast` with `torch.amp.autocast('cuda')`. |
| Slow generation | Use `--use_kv_cache 1` and run on GPU. |

## 📂 Directory layout
```
data/
 ├─ corpus/       # cleaned + split text data
 ├─ tokenizer/    # tokenizer artifacts
src/
 ├─ data_extractor/
 ├─ tokenizer/
 ├─ llm/          # GPT core (gpt.py, layers, rope, moe, rmsnorm, kvcache)
 ├─ train/        # training loop (train.py)
 └─ test/         # evaluation scripts
runs/
 └─ <experiment>/ # checkpoints + logs + samples
test.py           # quick inference script
```

## Artifacts
| Type | Location |
| --- | --- |
| Tokenizer | `data/tokenizer/tokenizer.json`, `vocab.json`, `merges.txt` |
| Checkpoints | `runs/<exp>/model_best.pt`, `model_final.pt` |
| Samples | Printed or saved under run directory |
| Logs | CLI progress with loss / aux_loss / sample text |

## Example metrics (small run)
| Step | main_loss | aux_loss | val_loss | Sample |
| --- | --- | --- | --- | --- |
| 200 | 5.06 | 6.04 | 4.61 | noisy text |
| 1000 | 3.40 | 6.01 | 3.57 | structured phrases |
| 4000 | 2.95 | 1.87 | 3.10 | coherent HPC sentences |

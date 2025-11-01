import trafilatura
from pathlib import Path

OUT_PATH = Path("data/corpus/hpc_corpus_extra.txt")
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

urls = [
    # --- CUDA / GPU programming ---
    "https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html",
    "https://developer.nvidia.com/blog/tag/cuda/",
    "https://developer.nvidia.com/blog/optimizing-hpc-applications-using-cuda/",
    # --- OpenMP ---
    "https://openmp.org/specifications/",
    "https://hpc.llnl.gov/tuts/openMP/",
    "https://computing.llnl.gov/tutorials/openMP/",
    # --- MPI ---
    "https://mpi-forum.org/docs/",
    "https://hpc-tutorials.llnl.gov/mpi/",
    "https://pages.tacc.utexas.edu/~eijkhout/pcse/html/mpi.html",
    # --- HPC Architecture & Node-level topics ---
    "https://hpc-wiki.info/hpc/Node_level_parallelism",
    "https://hpc-wiki.info/hpc/NUMA",
    "https://hpc-wiki.info/hpc/Performance_analysis",
    "https://lumi-supercomputer.github.io/",
    "https://docs.lrz.de/",
    "https://pop-coe.eu/",
]

with open(OUT_PATH, "w", encoding="utf-8") as out:
    for url in urls:
        print(f"⏬ Fetching {url}")
        try:
            downloaded = trafilatura.fetch_url(url)
            text = trafilatura.extract(downloaded, include_comments=False)
            if text and len(text.split()) > 100:
                out.write("<|endoftext|>\n" + text.strip() + "\n\n")
                print(f"✅ Added {len(text.split())} words from {url}")
            else:
                print(f"⚠️ Skipped (too short): {url}")
        except Exception as e:
            print(f"❌ Failed {url} | {e}")

print(f"\n💾 Saved final merged corpus to: {OUT_PATH}")

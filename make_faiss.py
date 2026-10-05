# this is the efficient version of making the faiss index that uses the gpu to make the index and cpu to store it.

import faiss 
import numpy as np
import os
import time
import sys
import torch

BYTES_PER_FLOAT = 4  # float32 is always 4 bytes

# Performance / safety knobs
BATCH_VECTORS = 8192  # number of vectors to add per index.add(); tune for speed vs peak RAM
PRINT_EVERY_VECTORS = 20_000

def get_model_params(model_code: str):
    if model_code == "Q600M":
        model_type = "Qwen/Qwen3-Embedding-0.6B"
        embed_dim = 1024
    elif model_code == "Q4B":
        model_type = "Qwen/Qwen3-Embedding-4B"
        embed_dim = 2560
    elif model_code == "Q8B":
        model_type = "Qwen/Qwen3-Embedding-8B"
        embed_dim = 4096
    elif model_code == "H270M":
        model_type = "microsoft/harrier-oss-v1-270m"
        embed_dim = 640
    elif model_code == "H600M":
        model_type = "microsoft/harrier-oss-v1-0.6b"
        embed_dim = 1024
    elif model_code == "H27B":
        model_type = "microsoft/harrier-oss-v1-27b"
        embed_dim = 5376
    else:
        raise ValueError(f"Invalid model code: {model_code}")

    return model_type, embed_dim

def main():
    if len(sys.argv) < 4:
        print("Usage: python make_faiss.py <MODEL_CODE> <SENTENCES>")
        print("Available models: Q600M, Q4B, Q8B, H270M, H600M, H27B")
        sys.exit(1)

    model_code = sys.argv[1]
    sentences_arg = int(sys.argv[2])
    dataset = sys.argv[3]

    model_type, embed_dim = get_model_params(model_code)
    bytes_per_vector = embed_dim * BYTES_PER_FLOAT

    binary_path = f"embeddings/{dataset}/{model_type}/embeddings_{sentences_arg}.bin"
    faiss_index_path = f"faiss/{dataset}/{model_type}/FAISS_{model_code}_{sentences_arg}.faiss"

    if not os.path.isfile(binary_path):
        raise FileNotFoundError(f"Missing embeddings file: {binary_path}")

    file_size = os.path.getsize(binary_path)
    total_vectors, remainder = divmod(file_size, bytes_per_vector)
    if remainder != 0:
        raise ValueError(
            f"{binary_path} size ({file_size} bytes) is not a multiple of BYTES_PER_VECTOR ({bytes_per_vector}). "
            f"Remainder={remainder} bytes. Check EMBED_DIM / dtype / file integrity."
        )

    print(
        f"Reading {binary_path}: {file_size / (1024**3):.2f} GiB, {total_vectors:,} vectors "
        f"(dim={embed_dim}, dtype=float32)."
    )

    mmap = np.memmap(binary_path, dtype=np.float32, mode="r", shape=(total_vectors, embed_dim))

    cpu_index = faiss.IndexFlatIP(embed_dim)
    res = faiss.StandardGpuResources()  # use a single GPU
    gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)  # move to GPU 0

    added = 0
    last_printed = 0
    started = time.time()

    for start in range(0, total_vectors, BATCH_VECTORS):
        end = min(start + BATCH_VECTORS, total_vectors)
        batch = np.ascontiguousarray(mmap[start:end], dtype=np.float32)
        gpu_index.add(batch)
        added = end

        if (added - last_printed >= PRINT_EVERY_VECTORS) or added == total_vectors:
            last_printed = added
            elapsed = max(time.time() - started, 1e-9)
            rate = added / elapsed
            remaining = max(total_vectors - added, 0)
            eta_s = remaining / max(rate, 1e-9)
            print(
                f"{added:,}/{total_vectors:,} vectors added "
                f"({rate:,.0f} vec/s, ETA {eta_s/60:.1f} min)"
            )
            free, total = torch.cuda.mem_get_info()
            used = (total - free) / 1024**3
            print(f"{added:,}: {used:.2f} GiB used")

    if added != total_vectors:
        raise RuntimeError(f"Expected to add {total_vectors:,} vectors, but added {added:,}.")

    os.makedirs(os.path.dirname(faiss_index_path), exist_ok=True)
    cpu_index = faiss.index_gpu_to_cpu(gpu_index)
    faiss.write_index(cpu_index, faiss_index_path)
    print(f"Wrote index: {faiss_index_path}")

    return 0


if __name__ == "__main__":
    main()


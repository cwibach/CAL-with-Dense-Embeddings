import array
import csv
import gc
import math
import os
import sys
import tarfile
import time
from collections import deque

import torch
from chunklet import Chunklet
from dotenv import load_dotenv
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, BitsAndBytesConfig

DTYPE_CODE = "f"  # float32

# --- Defaults (can be edited) ---
TGZ_OVERRIDE = None
OVERLAP = 50
DECODE_ERRORS = "ignore"
CLEAR_CUDA_EVERY = 10


def dataset_to_tgz(dataset: str) -> str:
    if dataset == "athome1":
        return "tgz_files/athome1mu4x7.tgz"
    if dataset == "athome2":
        return "tgz_files/athome2vQi9o.tgz"
    if dataset == "athome3":
        return "tgz_files/athome3sXaWM.tgz"
    if dataset == "athome4":
        return "tgz_files/athome4b5Qz8.tgz"
    raise ValueError(f"Unknown dataset: {dataset}")


def get_docno(doc_title: str) -> str:
    return doc_title.split("/")[-1]


def maybe_clear_cuda_cache(batch_counter: int, clear_every_n: int) -> None:
    if (
        torch is not None
        and torch.cuda.is_available()
        and clear_every_n > 0
        and (batch_counter % clear_every_n == 0)
    ):
        torch.cuda.empty_cache()


def get_model_params(model_code: str):
    if model_code == "Q600M":
        model_type = "Qwen/Qwen3-Embedding-0.6B"
        precision = "fp16"
        outer_batch_size = 8
    elif model_code == "Q4B":
        model_type = "Qwen/Qwen3-Embedding-4B"
        precision = "fp16"
        outer_batch_size = 4
    elif model_code == "Q8B":
        model_type = "Qwen/Qwen3-Embedding-8B"
        precision = "int8"
        outer_batch_size = 2
    elif model_code == "H270M":
        model_type = "microsoft/harrier-oss-v1-270m"
        precision = "fp16"
        outer_batch_size = 64
    elif model_code == "H600M":
        model_type = "microsoft/harrier-oss-v1-0.6b"
        precision = "fp16"
        outer_batch_size = 32
    elif model_code == "H27B":
        model_type = "microsoft/harrier-oss-v1-27b"
        precision = "4bit"
        outer_batch_size = 2
    else:
        raise ValueError(f"Invalid model code: {model_code}")

    return model_type, precision, outer_batch_size


def load_embedding_model(model_name: str, precision: str) -> SentenceTransformer:
    model_kwargs = {
        "device_map": "auto",
        "trust_remote_code": True,
        "attn_implementation": "flash_attention_2"
    }

    if precision == "fp16":
        model_kwargs["torch_dtype"] = torch.bfloat16
    elif precision == "4bit":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    elif precision == "int8":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        raise ValueError("precision must be: fp16 | int8 | 4bit")

    model = SentenceTransformer(
        model_name,
        backend="torch",
        model_kwargs=model_kwargs,
        processor_kwargs={"use_fast": True},
    )

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    model.precision_mode = precision

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return model


def embed_batch(sections: list[str], model: SentenceTransformer, precision_mode: str):
    if precision_mode in {"fp16", "bf16"} and torch.cuda.is_available():
        dtype = torch.bfloat16
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            emb = model.encode(sections, convert_to_numpy=True, batch_size=len(sections))
    else:
        with torch.inference_mode():
            emb = model.encode(sections, convert_to_numpy=True, batch_size=len(sections))
    return emb


def run_for_sentences(
    tgz_path: str,
    tokenizer,
    model: SentenceTransformer,
    sentences: int,
    overlap: int,
    outer_batch_size: int,
    clear_cuda_every: int,
    out_dir: str,
) -> None:
    
    def token_counter(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    chunker = Chunklet(token_counter=token_counter)

    tokens_cap = max(sentences * 20, 10000)
    batch_size = outer_batch_size

    os.makedirs(out_dir, exist_ok=True)
    bin_path = os.path.join(out_dir, f"embeddings_{sentences}.bin")
    csv_path = os.path.join(out_dir, f"chunks_{sentences}.csv")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    total_chunks = 0
    to_embed: deque[str] = deque()
    batch_counter = 0
    doc_count = 0
    embedding_id = 0

    print(f"[{sentences} sentences] Saving to {bin_path} and {csv_path}")

    t0 = time.perf_counter()

    with tarfile.open(tgz_path, "r:gz") as tar, \
         open(bin_path, "wb") as bin_fp, \
         open(csv_path, "w", newline="") as chunks_fp:
        
        chunks_writer = csv.writer(chunks_fp)
        
        for member in tar:
            if not member.isfile():
                continue

            docno = get_docno(member.name)
            
            f = tar.extractfile(member)
            if f is None:
                continue
                
            raw = f.read()
            f.close()
            text = raw.decode("utf-8", errors=DECODE_ERRORS)

            # Note exactly where this doc starts
            chunks_writer.writerow([docno, embedding_id])

            sections = chunker.chunk(
                text=text,
                mode="hybrid",
                max_sentences=sentences,
                max_tokens=tokens_cap,
                overlap_percent=overlap,
                lang="en",
            )
            
            total_chunks += len(sections)
            to_embed.extend(sections)
            embedding_id += len(sections)
            
            while len(to_embed) >= batch_size:
                batch = [to_embed.popleft() for _ in range(batch_size)]
                batch_counter += 1
                emb = embed_batch(batch, model, getattr(model, "precision_mode", "fp32"))

                for e in emb:
                    arr = array.array(DTYPE_CODE, e)
                    arr.tofile(bin_fp)

                maybe_clear_cuda_cache(batch_counter, clear_cuda_every)
                
            doc_count += 1

        # Flush remainder
        while len(to_embed) > 0:
            n = min(batch_size, len(to_embed))
            batch = [to_embed.popleft() for _ in range(n)]
            batch_counter += 1
            emb = embed_batch(batch, model, getattr(model, "precision_mode", "fp32"))

            for e in emb:
                arr = array.array(DTYPE_CODE, e)
                arr.tofile(bin_fp)

            maybe_clear_cuda_cache(batch_counter, clear_cuda_every)

    wall_seconds = time.perf_counter() - t0
    docs_per_sec = doc_count / wall_seconds if wall_seconds > 0 else 0.0
    
    print(f"[{sentences} sentences] Processed {doc_count} docs (Total chunks: {total_chunks}) "
          f"in {wall_seconds:.2f}s ({docs_per_sec:.2f} docs/sec)")


def main():
    load_dotenv()
            
    if len(sys.argv) < 4:
        print("Usage: python encode_new.py <MODEL_CODE> <SENTENCES>")
        print("Available models: Q600M, Q4B, Q8B, H270M, H600M, H27B")
        sys.exit(1)
        
    model_code = sys.argv[1]
    sentences_arg = int(sys.argv[2])
    dataset = sys.argv[3]

    tgz_path = TGZ_OVERRIDE or dataset_to_tgz(dataset)
    model_type, precision, outer_batch_size = get_model_params(model_code)

    print(f"=== Encoding Dataset: {dataset} ===")
    print(f"Tar archive: {tgz_path}")
    print(f"Model: {model_code} ({model_type})")
    print(f"Precision: {precision}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_type, trust_remote_code=True)
    model = load_embedding_model(model_type, precision)
    
    out_dir = f"embeddings/{dataset}/{model_type}"
    print(f"Outputs will be written to: {out_dir}")

    print(f"\n--- Starting embed pass for max_sentences = {sentences_arg} ---")
    run_for_sentences(
        tgz_path=tgz_path,
        tokenizer=tokenizer,
        model=model,
        sentences=sentences_arg,
        overlap=OVERLAP,
        outer_batch_size=outer_batch_size,
        clear_cuda_every=CLEAR_CUDA_EVERY,
        out_dir=out_dir,
    )

if __name__ == "__main__":
    main()

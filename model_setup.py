import array
import csv
import gc
import os
import random
import resource
import tarfile
import time
from collections import deque
from dataclasses import dataclass
import math

import torch
from chunklet import Chunklet
from dotenv import load_dotenv
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from sentence_transformers.sentence_transformer.modules import Pooling, Transformer
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModel

load_dotenv()
api_key = os.getenv("HUGGINGFACE_TOKEN")
login()

models = [
    {"name": "Qwen/Qwen3-Embedding-0.6B", "quantization": None},
    {"name": "Qwen/Qwen3-Embedding-4B", "quantization": None},
    {"name": "Qwen/Qwen3-Embedding-8B", "quantization": None},
    {"name": "microsoft/harrier-oss-v1-270m", "quantization": None},
    {"name": "microsoft/harrier-oss-v1-0.6b", "quantization": None},
    {"name": "microsoft/harrier-oss-v1-27b", "quantization": "4bit"},
]

for spec in models:
    model_name = spec["name"]
    AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if spec["quantization"] == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            device_map="auto",
            quantization_config=quantization_config,
        )
    else:
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
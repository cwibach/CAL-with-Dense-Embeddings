import tarfile
from sentence_transformers import SentenceTransformer
import numpy as np
import csv
import gc
import torch
from sentence_transformers import SentenceTransformer
from transformers import BitsAndBytesConfig

datasets = ['athome1', 'athome2', 'athome3']
MODELS = ['Q600M', 'Q4B', 'Q8B', 'H600M', 'H270M']
topic_options = [1,2,3]
INCLUDE_DESC = True

def get_model_params(model_code: str):
    if model_code == "Q600M":
        model_type = "Qwen/Qwen3-Embedding-0.6B"
        precision = "fp16"
    elif model_code == "Q4B":
        model_type = "Qwen/Qwen3-Embedding-4B"
        precision = "fp16"
    elif model_code == "Q8B":
        model_type = "Qwen/Qwen3-Embedding-8B"
        precision = "int8"
    elif model_code == "H270M":
        model_type = "microsoft/harrier-oss-v1-270m"
        precision = "fp16"
    elif model_code == "H600M":
        model_type = "microsoft/harrier-oss-v1-0.6b"
        precision = "fp16"
    elif model_code == "H27B":
        model_type = "microsoft/harrier-oss-v1-27b"
        precision = "4bit"
    else:
        raise ValueError(f"Invalid model code: {model_code}")

    return model_type, precision

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

def main():
    for dataset in datasets:
        if dataset in ['athome1', 'athome2', 'athome3']:
            topics_path = "tgz_files/tr-2015-topics-judgments.tgz"
            topic_file = "topics"
        elif dataset == "athome4":
            topics_path = "tgz_files/tr-2016-topics-judgments.tgz"
            topic_file = "tr2016-ext-topics.txt"
        else:
            print(f"Invalid dataset: {dataset}")
            pass

        for model_code in MODELS:

            model_type, precision = get_model_params(model_code)
            model = load_embedding_model(model_type, precision)

            tar = tarfile.open(topics_path, "r:gz")
            topics = tar.extractfile(topic_file).read().decode('utf-8','replace')

            if dataset in ['athome1', 'athome2', 'athome3']:
                write_path = f"topics/topics_{dataset}_{model_code}.csv"
                f = open(write_path, 'a', newline="")
                embedding_writer = csv.writer(f)
                for topic in topics.splitlines():
                    topid, description = topic.split("\t")
                    if dataset in topid:
                        emb = model.encode(description, convert_to_numpy=True)
                        topNum = int(topid[6:])
                        rowData = np.insert(emb.tolist(), 0, topNum)
                        embedding_writer.writerow(rowData)

            elif dataset == "athome4":
                for mode in topic_options:
                    suffix = ""
                    if mode == 1:
                        suffix = "query_desc"
                    elif mode == 2:
                        suffix = "desc"
                    elif mode == 3:
                        suffix = "query"

                    write_path = f"topics/topics_{dataset}_{model_code}_{suffix}.csv"
                    f = open(write_path, 'a', newline="")
                    embedding_writer = csv.writer(f)
                    for topic in topics.splitlines():
                        """
                          Sample topic format:
                           401  Summer Olympics -- All documents concerning a bid to host the Summer Olympic Games in Florida.
                        """
                        halves = topic.split("--")
                        topNum = halves[0][0:3]

                        description = f"{halves[0][3:].strip()} {halves[1].strip()}" # Default
                        if mode == 1:
                            description = f"{halves[0][3:].strip()} {halves[1].strip()}" # Query & Description
                        elif mode == 2:
                            description = description = f"{halves[1].strip()}" # Just Description
                        elif mode == 3:
                            description = f"{halves[0][3:].strip()}" # Just Query

                        print(f"{mode} - {description}")
                        emb = model.encode(description, convert_to_numpy=True)
                        rowData = np.insert(emb.tolist(), 0, topNum)
                        embedding_writer.writerow(rowData)

            del tar, topics, model
            f.close()

            print(f"Done {dataset} {model_code}")

if __name__ == "__main__":
    main()

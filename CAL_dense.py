import faiss
import csv
import numpy as np
import faulthandler
import os
import math
import sys
from sklearn.linear_model import SGDClassifier
from CAL_Functions import load_topics, load_sections, docno_from_section, sections_from_docno, make_relevancy_dict, random_indices, load_faiss_index, reconstruct_vectors, stack_vectors, select_candidates, remake_faiss_index

K = 1
MAX_EFFORT = 2
MAX_ITER=1000
RANDOM_SAMPLE = 100
SEARCH_START_K = 64

def dataset_to_qrels(dataset):
    if dataset == "athome1":
        return "tgz_files/tr-2015-topics-judgments.tgz"
    if dataset == "athome2":
        return "tgz_files/tr-2015-topics-judgments.tgz"
    if dataset == "athome3":
        return "tgz_files/tr-2015-topics-judgments.tgz"
    if dataset == "athome4":
        return "tgz_files/tr-2016-topics-judgments.tgz"
    raise ValueError(f"Unknown dataset: {dataset}")

def get_model_params(model_code: str):
    if model_code == "Q600M":
        model_type = "Qwen/Qwen3-Embedding-0.6B"
    elif model_code == "Q4B":
        model_type = "Qwen/Qwen3-Embedding-4B"
    elif model_code == "Q8B":
        model_type = "Qwen/Qwen3-Embedding-8B"
    elif model_code == "H270M":
        model_type = "microsoft/harrier-oss-v1-270m"
    elif model_code == "H600M":
        model_type = "microsoft/harrier-oss-v1-0.6b"
    elif model_code == "H27B":
        model_type = "microsoft/harrier-oss-v1-27b"
    else:
        raise ValueError(f"Invalid model code: {model_code}")

    return model_type

def main():
    faulthandler.enable()

    if len(sys.argv) < 4:
        print("Usage: python encode_new.py <MODEL_CODE> <SENTENCES> <TRAINDATA>")
        print("Available models: Q600M, Q4B, Q8B, H270M, H600M, H27B")
        print("Available sentences: 32, 64, 128, 256, 512, 1024")
        print("Traindata options: JudgedChunks, AllChunks")
        sys.exit(1)
    
    model_code = sys.argv[1]
    sentences = int(sys.argv[2])
    train_mode = sys.argv[3].strip()
    dataset = sys.argv[4].strip()

    if train_mode not in ["JudgedChunks", "AllChunks"]:
        print("Invalid trainmode")
        print("Traindata options: JudgedChunks, AllChunks")
        sys.exit(1)

    qrels_path = dataset_to_qrels(dataset)
    model_type = get_model_params(model_code)
    faiss_path = os.path.join("faiss", dataset, model_type, f"FAISS_{model_code}_{sentences}.faiss")
    chunks_path = os.path.join("embeddings", dataset, model_type, f"chunks_{sentences}.csv")
    topics_path = os.path.join("topics", f"topics_{dataset}_{model_code}2.csv")
    write_path = os.path.join("results2", dataset, model_type, f"{train_mode}_{sentences}.csv")

    use_non_relevant = (dataset == 'athome4')
    if use_non_relevant:
        relevancies, non_relevancies = make_relevancy_dict(dataset, qrels_path, True)
    else:
        relevancies = make_relevancy_dict(dataset, qrels_path, False)

    search_index, reconstruct_index = load_faiss_index(faiss_path)
    need_new_index = False
    print("FAISS index loaded")
    starts, docnos = load_sections(chunks_path, search_index.ntotal)
    total_sections = starts[-1] - 1
    docno_to_index = {docno: i for i, docno in enumerate(docnos)}
    
    topic_embeds = load_topics(topics_path)
    topic_embeds = {topic: topic_embeds[topic] for topic in relevancies.keys()}
    print("document & topic Chunks Loaded")

    forward_index = {chunk: i for i, chunk in enumerate(range(search_index.ntotal))}
    reverse_index = {i: chunk for chunk, i in forward_index.items()}

    os.makedirs(os.path.dirname(write_path), exist_ok=True)

    topics = topic_embeds.keys()
    with open(write_path, "w", newline="") as out_file:
        writer = csv.writer(out_file)

        for topic in topics:
            if need_new_index:
                search_index, reconstruct_index = load_faiss_index(faiss_path)
                forward_index = {chunk: i for i, chunk in enumerate(range(search_index.ntotal))}
                reverse_index = {i: chunk for chunk, i in forward_index.items()}
                need_new_index = False

            relevant = relevancies[topic]
            if use_non_relevant:
                non_relevant = non_relevancies[topic]
            num_relevant = len(relevant)

            model = SGDClassifier(
                loss="log_loss",
                penalty='l2',
                alpha=1e-4,
                max_iter=MAX_ITER,
                tol=1e-3,
                learning_rate="optimal",
                fit_intercept=True,
                shuffle=True,
                average=False,
                warm_start=True,
                random_state=0
            )

            if num_relevant == 0:
                continue

            topic_vec = np.asarray(topic_embeds[topic], dtype=np.float32)
            if topic_vec.shape[0] != search_index.d:
                raise ValueError(
                    f"Topic embedding dim {topic_vec.shape[0]} does not match index dim {search_index.d}"
                )

            pos_vectors = [topic_vec]
            neg_vectors = []
            avoid = set()
            judged = 0
            max_judged = math.ceil(num_relevant * MAX_EFFORT)

            print(f"Beginning topic {topic} with {num_relevant} relevant docs")

            while judged < max_judged:
                if len(avoid) >= 2000:
                    search_index, forward_index, reverse_index = remake_faiss_index(search_index, avoid, forward_index)
                    avoid = set()
                    need_new_index = True

                available = search_index.ntotal
                if available <= 0:
                    break

                rand_count = min(RANDOM_SAMPLE, available)
                
                rand_indices = random_indices(len(forward_index) - 1, avoid, rand_count)
                rand_vectors = reconstruct_vectors(search_index, list(rand_indices))

                pos_matrix = stack_vectors(pos_vectors, search_index.d)
                neg_matrix = stack_vectors(neg_vectors, search_index.d)

                X_train = np.vstack([pos_matrix, neg_matrix, rand_vectors])
                y_train = np.concatenate(
                    [
                        np.ones(pos_matrix.shape[0], dtype=np.int32),
                        np.zeros(neg_matrix.shape[0] + rand_vectors.shape[0], dtype=np.int32),
                    ]
                )

                model.fit(X_train, y_train)

                weights = np.asarray(model.coef_, dtype=np.float32).reshape(1, -1)
                query = np.ascontiguousarray(weights, dtype=np.float32)

                candidate_indices = select_candidates(search_index, query, avoid, K, SEARCH_START_K)
                if not candidate_indices:
                    break
                
                for new_candidate_idx in candidate_indices:
                    assert new_candidate_idx in forward_index, (
                                f"FAISS returned {new_candidate_idx}, "
                                f"but map only contains 0-{len(forward_index)-1}"
                                F"and index contains {search_index.ntotal} vectors"
                            )
                    docno = docno_from_section(starts, docnos, forward_index[new_candidate_idx])
                    is_relevant = docno in relevant

                    if not use_non_relevant:
                        is_non_relevant = True
                    else:
                        is_non_relevant = (docno in non_relevant)

                    if is_relevant:
                        writer.writerow([topic, docno, 1])
                    elif is_non_relevant:
                        writer.writerow([topic, docno, 0])
                    else: # unjudged and using non_relevant
                        writer.writerow([topic, docno, -1])
                    
                    judged += 1
                    old_section_indices = sections_from_docno(starts, docnos, docno, docno_to_index)
                    doc_sections = list(reverse_index[i] for i in old_section_indices)
                    if doc_sections:
                        avoid.update(doc_sections)
                    else:
                        avoid.add(new_candidate_idx)

                    if train_mode == "AllChunks" and doc_sections:
                        section_vectors = reconstruct_vectors(reconstruct_index, old_section_indices)
                        if is_relevant:
                            pos_vectors.extend(section_vectors)
                        else: # trains on unjudged too
                            neg_vectors.extend(section_vectors)
                    else:
                        candidate_vec = reconstruct_vectors(reconstruct_index, [forward_index[new_candidate_idx]])
                        if candidate_vec.shape[0] == 0:
                            continue
                        if is_relevant:
                            pos_vectors.append(candidate_vec[0])
                        else: # trains on unjudged too
                            neg_vectors.append(candidate_vec[0])

                    if judged >= max_judged:
                        break


if __name__ == "__main__":
    main()

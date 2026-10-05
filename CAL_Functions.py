import faiss
import os
import csv
import bisect
import numpy as np
import tarfile
from random import randint

def random_indices(max_index, exclude, count=100):
    """
    Docstring for random_indices
    
    :param max_index: maximum value for random index
    :param exclude: set of indices to avoid choosing (must be much smaller than max)
    :param count: how many random indices to find

    returns set of random indices to use for training
    """
    indices = set()
    while len(indices) < count:
        r = randint(0,max_index-1)
        if r not in exclude:
            indices.add(r)

    return indices

def make_relevancy_dict(dataset, qrels_path, include_non_relevant=True):
    relevancies = {}
    if include_non_relevant:
        non_relevant = {}

    with tarfile.open(qrels_path, 'r:gz') as tar:
        if "tr-2016-topics-judgments.tgz" in qrels_path:
            qrels_file = tar.extractfile('athome4.facetsandqrels')
            if qrels_file is None:
                raise FileNotFoundError("Missing athome4.facetsandqrels in qrels tar.")
            with qrels_file:
                qrels = qrels_file.read().decode('utf-8', 'replace')
            
                
            for line in qrels.splitlines():
                line_parts = line.split()

                if len(line_parts) == 4:
                    topid = line_parts[0]
                    docno = line_parts[1]
                    rel = int(line_parts[2])

                    if (rel >= 1):
                        if topid in relevancies:
                            relevancies[topid].add(docno)
                        else:
                            relevancies[topid] = {docno}
                    elif (rel == 0) and include_non_relevant:
                        if topid in non_relevant:
                            non_relevant[topid].add(docno)
                        else:
                            non_relevant[topid] = {docno}

        elif "tr-2015-topics-judgments.tgz" in qrels_path:
            index = {i.name: i for i in tar.getmembers()}
            for key in index.keys():
                if dataset in key:
                    body = tar.extractfile(key)
                    if body is not None:
                        text = body.read().decode('utf-8', 'replace')
                        for line in text.splitlines():
                            docno, topid, rel = line.split('\t')
                            if docno.strip() == "docid":
                                continue

                            rel = int(rel)
                            topic = topid[6:]
                            if rel >= 1:
                                if topic in relevancies:
                                    relevancies[topic].add(docno)
                                else:
                                    relevancies[topic] = {docno}
                            elif rel < 1 and include_non_relevant:
                                if topid in non_relevant:
                                    non_relevant[topic].add(docno)
                                else:
                                    non_relevant[topic] = {docno}

        else:
            return None

    if include_non_relevant:
        return relevancies, non_relevant
    else:
        return relevancies

def load_topics(topics_path):
    topics = {}
    with open(topics_path, 'r', newline='') as file_data:
        csv_reader = csv.reader(file_data)
        for row in csv_reader:
            topic = str(int(float(row[0])))
            embedding = np.array([float(s) for s in row[1:]])
            topics[topic] = embedding
    return topics

def load_sections(sections_path, total_sections=None):
    """
    Docstring for load_sections
    
    :param sections_path: path to section docno mapping

    returns list of starting index for each document, and list of all docnos in same order
    """
    starts = []
    docnos = []
    with open(sections_path, 'r', newline='') as file_data:
        csv_reader = csv.reader(file_data)
        for row in csv_reader:
            docnos.append(row[0])
            starts.append(int(row[1]))

    if total_sections is None:
        total_sections = starts[-1] + 1
        
    starts.append(int(total_sections))
    return starts, docnos

def docno_from_section(starts, docnos, section_no):
    """
    Docstring for docno_from_section
    
    :param starts: list of starting indexes for docno sections
    :param docnos: list of all docnos in same order as starts
    :param section_no: section number to find docno of

    return docno containing appropriate section
    """
    i = bisect.bisect_right(starts, section_no) - 1
    if i < 0:
        return docnos[0]

    if i >= len(docnos):
        # print(f"Error, using last document. Section no {section_no} got index {i}")
        return docnos[-1]
    
    return docnos[i]

def sections_from_docno(starts, docnos, docno, docno_to_index=None):
    """
    Docstring for sections_from_docno
    
    :param starts: list of starting indexes for docno sections
    :param docnos: list of all docnos in same order as starts
    :param docno: docno to find all section numbers from

    return list of all sections that are in document
    """
    if docno_to_index is None:
        index = docnos.index(docno)
    else:
        index = docno_to_index.get(docno)
        if index is None:
            raise ValueError(f"Docno not found in mapping: {docno}")
    start_section = starts[index]
    end_section = starts[index + 1]

    sections = range(start_section, end_section)
    return sections

def load_faiss_index(faiss_path: str):
    if not os.path.isfile(faiss_path):
        raise FileNotFoundError(f"FAISS index not found: {faiss_path}")

    cpu_index = faiss.read_index(faiss_path)
    search_index = cpu_index

    gpu_count = faiss.get_num_gpus()
    if gpu_count > 0:
        res = faiss.StandardGpuResources()
        try:
            search_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
            print(f"FAISS GPU enabled (gpus={gpu_count}); using GPU 0 for search.")
        except Exception as exc:
            print(f"Warning: failed to move FAISS index to GPU ({exc}); using CPU.")
            search_index = cpu_index
    else:
        print("FAISS GPU not available; using CPU index.")

    return search_index, cpu_index

def reconstruct_vectors(index, ids):
    if not ids:
        return np.empty((0, index.d), dtype=np.float32)

    if isinstance(ids, range):
        start = ids.start
        count = len(ids)
        if count <= 0:
            return np.empty((0, index.d), dtype=np.float32)
        if hasattr(index, "reconstruct_n"):
            return np.asarray(index.reconstruct_n(int(start), int(count)), dtype=np.float32)

    if isinstance(ids, set):
        ids_list = sorted(ids)
    else:
        ids_list = list(ids)

    if not ids_list:
        return np.empty((0, index.d), dtype=np.float32)

    use_batch = hasattr(index, "reconstruct_n")
    is_sorted = all(ids_list[i] <= ids_list[i + 1] for i in range(len(ids_list) - 1))

    if use_batch and is_sorted:
        vectors = np.empty((len(ids_list), index.d), dtype=np.float32)
        run_start = ids_list[0]
        run_len = 1
        write_pos = 0

        for i in range(1, len(ids_list) + 1):
            if i < len(ids_list) and ids_list[i] == ids_list[i - 1] + 1:
                run_len += 1
                continue

            run_vectors = np.asarray(
                index.reconstruct_n(int(run_start), int(run_len)),
                dtype=np.float32
            )
            vectors[write_pos:write_pos + run_len] = run_vectors
            write_pos += run_len

            if i < len(ids_list):
                run_start = ids_list[i]
                run_len = 1

        return vectors

    vectors = np.empty((len(ids_list), index.d), dtype=np.float32)
    for i, idx in enumerate(ids_list):
        vectors[i] = index.reconstruct(int(idx))
    return vectors

def stack_vectors(vectors, dim):
    if not vectors:
        return np.empty((0, dim), dtype=np.float32)
    return np.ascontiguousarray(np.vstack(vectors), dtype=np.float32)

def select_candidates(search_index, query, avoid, k, start_k):
    if k <= 0:
        return []

    max_k = int(search_index.ntotal)
    if max_k <= 0:
        return []

    search_k = min(max(start_k, k), max_k)
    while True:
        _, indices = search_index.search(query, search_k)
        candidates = [int(idx) for idx in indices[0] if idx != -1 and idx not in avoid]
        if len(candidates) >= k or search_k >= max_k:
            return candidates[:k]
        search_k = min(search_k * 2, max_k)

def remake_faiss_index(old_index, ids_to_remove, old_fward_map):
    """
    Remakes a FAISS index by removing specified IDs.
    
    :param old_index: The original FAISS index.
    :param ids_to_remove: A set of IDs to remove from the index.
    :return: A new FAISS index and a new ID map.
    """
    ntotal = old_index.ntotal
    d = old_index.d
    
    ids_to_keep = [i for i in old_fward_map.keys() if i not in ids_to_remove]
    
    new_index = faiss.IndexFlatIP(d)
    
    # Batch reconstruct vectors to avoid reconstructing one by one
    vectors_to_add = old_index.reconstruct_n(0, ntotal)
    vectors_to_keep = vectors_to_add[ids_to_keep]
    
    if vectors_to_keep.shape[0] > 0:
        new_index.add(vectors_to_keep)

    gpu_count = faiss.get_num_gpus()
    if gpu_count > 0:
        res = faiss.StandardGpuResources()
        try:
            search_index = faiss.index_cpu_to_gpu(res, 0, new_index)
            print(f"FAISS GPU enabled (gpus={gpu_count}); using GPU 0 for search.")
        except Exception as exc:
            print(f"Warning: failed to move FAISS index to GPU ({exc}); using CPU.")
            search_index = new_index
    else:
        print("FAISS GPU not available; using CPU index.")
        
    new_id_map = {i: old_fward_map[j] for i, j in enumerate(ids_to_keep)}
    new_reverse_map = {original_id: i for i, original_id in new_id_map.items()}
    
    return search_index, new_id_map, new_reverse_map

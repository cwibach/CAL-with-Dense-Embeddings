import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from importlib.util import module_from_spec, spec_from_file_location
import math
import tarfile
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_DIR = REPO_ROOT / "prompt-for-total-recall" / "harness"
CLIENT_SPEC = spec_from_file_location("example_client", HARNESS_DIR / "example_client.py")
if CLIENT_SPEC is None or CLIENT_SPEC.loader is None:
    raise ImportError(f"Unable to load judge client from {HARNESS_DIR}")
CLIENT_MODULE = module_from_spec(CLIENT_SPEC)
CLIENT_SPEC.loader.exec_module(CLIENT_MODULE)
Judge = CLIENT_MODULE.Judge


MODEL = "gpt.oss.120b"
LOCAL_API = "http://localhost:8000/v1/chat/completions"
CORPORA_DIR = (REPO_ROOT / "../../corpora/total-recall").resolve()
TOPICS_ARCHIVE = CORPORA_DIR / "tr-2016-topics-judgments.tgz"
DOCUMENT_ARCHIVE = CORPORA_DIR / "athome4b5Qz8.tgz"
TOPIC_MEMBER = "tr2016-ext-topics.txt"
DOCUMENT_PREFIX = "athome4/"
ABRIDGED_DOCUMENT_MARKER = "\n\n[... middle of document omitted ...]\n\n"


def abridge_document(document: str) -> str:
    """Keep equal-length segments from the start and end of a failed request."""
    segment_length = len(document) // 4
    if segment_length == 0:
        return document
    return (
        document[:segment_length]
        + ABRIDGED_DOCUMENT_MARKER
        + document[-segment_length:]
    )


def load_topics(topics_archive: Path) -> dict[str, dict[str, str]]:
    """Return athome4 topics keyed by topic number."""
    with tarfile.open(topics_archive, "r:gz") as archive:
        topic_file = archive.extractfile(TOPIC_MEMBER)
        if topic_file is None:
            raise FileNotFoundError(f"Missing {TOPIC_MEMBER} in {topics_archive}")

        topics = {}
        for line in topic_file.read().decode("utf-8", "replace").splitlines():
            if not line.strip():
                continue
            topic_number, topic_text = line.split(None, 1)
            query, separator, description = topic_text.partition("--")
            if not separator:
                raise ValueError(f"Invalid topic line: {line!r}")
            topics[topic_number] = {
                "query": query.strip(),
                "description": description.strip(),
            }
    return topics


def read_documents(input_path: Path) -> list[tuple[str, str, str]]:
    """Read topic, document, and original judgment from the headerless CSV."""
    documents = []
    with input_path.open("r", newline="", encoding="utf-8") as input_file:
        for row_number, row in enumerate(csv.reader(input_file), start=1):
            if not row or not any(field.strip() for field in row):
                continue
            if len(row) != 3:
                raise ValueError(f"{input_path}:{row_number}: expected 3 columns")
            documents.append((row[0].strip(), row[1].strip(), row[2].strip()))
    return documents


def read_cache(cache_path: Path) -> dict[str, dict[str, int]]:
    if not cache_path.exists():
        return {}
    with cache_path.open("r", encoding="utf-8") as cache_file:
        return json.load(cache_file)


def save_cache(cache_path: Path, judgements: dict[str, dict[str, int]]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = cache_path.with_suffix(".tmp")
    with temporary_path.open("w", encoding="utf-8") as cache_file:
        json.dump(judgements, cache_file, indent=2, sort_keys=True)
    temporary_path.replace(cache_path)


def document_member(docno: str) -> str:
    return f"{DOCUMENT_PREFIX}{docno[:3]}/{docno}"


def read_document(archive: tarfile.TarFile, docno: str) -> str:
    document_file = archive.extractfile(document_member(docno))
    if document_file is None:
        raise FileNotFoundError(f"Document {docno} is not in the athome4 archive")
    return document_file.read().decode("utf-8", "replace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge athome4 documents with an LLM.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input CSV file, or a folder containing files to process recursively.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV file, or the output folder when --input is a folder.",
    )
    parser.add_argument(
        "--timings",
        type=Path,
        default=REPO_ROOT / "athome/llm_judgement_times.csv",
        help="CSV file for per-LLM-call timings and the final program duration.",
    )
    parser.add_argument(
        "--cache",
        type=Path,
        default=REPO_ROOT / "athome/cached_judgements" / MODEL / "athome4.json",
    )
    parser.add_argument("--topics", type=Path, default=TOPICS_ARCHIVE)
    parser.add_argument("--documents", type=Path, default=DOCUMENT_ARCHIVE)
    parser.add_argument("--model", default=MODEL, help="Name of the model exposed by the local server.")
    parser.add_argument("--api", default=LOCAL_API, help="OpenAI-compatible chat completions URL.")
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Maximum simultaneous LLM requests; cannot exceed 8 (default: 8).",
    )
    parser.add_argument(
        "--topic-percentage",
        "--topic-percent",
        type=float,
        default=100.0,
        help=(
            "Percentage of each topic's input rows to process from the beginning "
            "of that topic (default: 100)."
        ),
    )
    parser.add_argument(
        "--judge-all",
        action="store_true",
        help="Judge every input row instead of only rows whose original judgment is -1.",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=25,
        help="Save the cache after this many new judgments (default: 25).",
    )
    return parser.parse_args()


def main() -> int:
    program_started = time.perf_counter()
    args = parse_args()
    if args.save_every < 1:
        raise ValueError("--save-every must be at least 1")
    if not 1 <= args.workers <= 25:
        raise ValueError("--workers must be between 1 and 8")
    if not 0 <= args.topic_percentage <= 100:
        raise ValueError("--topic-percentage must be between 0 and 100")

    if not args.input.exists():
        raise FileNotFoundError(f"Input path does not exist: {args.input}")
    if args.input.is_dir():
        input_files = sorted(path for path in args.input.rglob("*") if path.is_file())
        if not input_files:
            raise FileNotFoundError(f"No input files found under {args.input}")
        if args.output.exists() and args.output.is_file():
            raise ValueError("--output must be a folder when --input is a folder")
    else:
        input_files = [args.input]

    topics = load_topics(args.topics)
    cached_judgements = read_cache(args.cache)
    judge = Judge(api=args.api, model=args.model)
    args.timings.parent.mkdir(parents=True, exist_ok=True)
    total_new_judgements = 0
    with args.timings.open("w", newline="", encoding="utf-8") as timings_file:
        timing_writer = csv.writer(timings_file)
        timing_writer.writerow(
            [
                "record_type",
                "topic",
                "docno",
                "judgement_seconds",
                "elapsed_program_seconds",
                "judgements_completed",
            ]
        )

        for input_path in input_files:
            file_started = time.perf_counter()
            documents = read_documents(input_path)
            unknown_topics = {topic for topic, _, _ in documents if topic not in topics}
            if unknown_topics:
                raise KeyError(f"Topics not present in {args.topics}: {sorted(unknown_topics)}")
            topic_totals = {}
            for topic, _, _ in documents:
                topic_totals[topic] = topic_totals.get(topic, 0) + 1
            topic_limits = {
                topic: math.ceil(total * args.topic_percentage / 100)
                for topic, total in topic_totals.items()
            }

            results = []
            new_judgements = 0
            topic_counts = {}
            pending = []
            pending_originals = {}

            def judge_one(item):
                topic, docno, topic_info, document = item
                started = time.perf_counter()
                response = judge.raw(topic_info["query"], topic_info["description"], document)
                if response is not None and response.get("finish_reason") == "length" and len(document) > 1:
                    response = judge.raw(
                        topic_info["query"], topic_info["description"], abridge_document(document)
                    )
                return topic, docno, topic_info, document, response, time.perf_counter() - started

            def complete_pending() -> None:
                nonlocal new_judgements
                if not pending:
                    return

                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    completed = executor.map(judge_one, pending)
                    retry_items = []
                    for topic, docno, topic_info, document, raw_response, judgement_seconds in completed:
                        if raw_response is not None and raw_response.get("finish_reason") == "length":
                            if len(document) > 1:
                                retry_items.append((topic, docno, topic_info, abridge_document(document)))
                                continue
                            raise RuntimeError(f"Judge hit max_tokens for {topic}/{docno}")
                        if raw_response is None:
                            print(
                                f"No response from local judge for {topic}/{docno}; skipping document.",
                                flush=True,
                            )
                            continue
                        judgement = judge.parse(raw_response["content"])
                        if judgement is None:
                            raise RuntimeError(
                                f"No RELEVANT/NOT RELEVANT verdict for {topic}/{docno}: "
                                f"{raw_response['content']!r}"
                            )
                        cached_judgements.setdefault(topic, {})[docno] = judgement
                        new_judgements += 1
                        timing_writer.writerow(
                            [
                                "judgement",
                                topic,
                                docno,
                                f"{judgement_seconds:.6f}",
                                f"{time.perf_counter() - program_started:.6f}",
                                total_new_judgements + new_judgements,
                            ]
                        )
                        timings_file.flush()
                        if (total_new_judgements + new_judgements) % args.save_every == 0:
                            save_cache(args.cache, cached_judgements)
                        results.append((topic, docno, judgement, pending_originals[(topic, docno)]))
                if retry_items:
                    pending[:] = retry_items
                    return
                pending.clear()
                pending_originals.clear()

            try:
                with tarfile.open(args.documents, "r:gz") as document_archive:
                    for topic, docno, original_judgement in documents:
                        topic_count = topic_counts.get(topic, 0)
                        if topic_count >= topic_limits[topic]:
                            continue
                        topic_counts[topic] = topic_count + 1

                        cached = cached_judgements.get(topic, {}).get(docno)
                        should_judge = args.judge_all or original_judgement == "-1"
                        if cached is not None and should_judge:
                            judgement = cached
                        elif not should_judge:
                            judgement = int(original_judgement)
                        else:
                            document = read_document(document_archive, docno)
                            topic_info = topics[topic]
                            pending.append((topic, docno, topic_info, document))
                            pending_originals[(topic, docno)] = original_judgement
                            if len(pending) == args.workers:
                                complete_pending()
                            continue
                        results.append((topic, docno, judgement, original_judgement))
                    complete_pending()
            finally:
                if pending:
                    complete_pending()
                if new_judgements:
                    save_cache(args.cache, cached_judgements)

            relative_path = input_path if args.input.is_file() else input_path.relative_to(args.input)
            output_path = args.output if args.input.is_file() else args.output / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
            with temporary_output.open("w", newline="", encoding="utf-8") as output_file:
                csv.writer(output_file).writerows(results)
            temporary_output.replace(output_path)
            total_new_judgements += new_judgements
            file_seconds = time.perf_counter() - file_started
            timing_writer.writerow(
                ["file_summary", "", str(relative_path), "", f"{file_seconds:.6f}", new_judgements]
            )
            timings_file.flush()
            print(
                f"Completed {input_path} -> {output_path}: {len(results)} result rows, "
                f"{new_judgements} LLM judgments in {file_seconds:.2f} seconds.",
                flush=True,
            )

    if total_new_judgements:
        save_cache(args.cache, cached_judgements)

    print(
        f"Completed {len(input_files)} file(s), {total_new_judgements} LLM judgments in "
        f"{time.perf_counter() - program_started:.2f} seconds."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
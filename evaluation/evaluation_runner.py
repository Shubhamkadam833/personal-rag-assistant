"""
evaluation_runner.py — measures retrieval quality on the synthetic benchmark.

Run:   .venv\\Scripts\\python -m evaluation.evaluation_runner [--name baseline]

WHAT IT DOES:
  1. Generates the synthetic PDFs into a temp folder.
  2. Indexes them through the real pipeline (extract -> chunk -> sanitize ->
     embed -> store) into a SEPARATE temp index. Your real data/ folder and
     documents are never read or written.
  3. Runs every answerable question through the real retrieve() path and
     scores the results (see retrieval_metrics.py).
  4. Writes a results JSON with the config, model, dataset version and git
     commit, so later phases can be compared against it.

Retrieval only: no Claude API calls, so it costs nothing and is repeatable.
Questions are asked without chat history, so the query rewriter never runs.
"""

import argparse
import hashlib
import json
import statistics
# Only used for a fixed `git rev-parse` call with no user input.
import subprocess  # nosec B404
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from evaluation.corpus import CORPUS_VERSION, build_corpus
from evaluation.retrieval_metrics import evidence_intact, percentile, recall_at_k, reciprocal_rank
from rag import document_registry, vector_store
from rag.embeddings import MODEL_NAME, embed_texts
from rag.pdf_loader import load_pdf
from rag.retriever import retrieve
from rag.text_chunker import chunk_pages
from security.prompt_guard import sanitize_chunk_text

ROOT = Path(__file__).parent
DATASET = ROOT / "benchmark_dataset.json"
RESULTS_DIR = ROOT / "results"

CHUNK_SIZE = 150
OVERLAP = 30


def _isolate_storage(store_dir: Path) -> None:
    """Point the vector store and registry at a temp folder for this run.

    Shortcut: these modules hard-code their storage paths, so the runner
    overrides the module constants (the same approach the tests use). A
    central config in Phase 2 should replace this.
    """
    vector_store.STORE_DIR = store_dir
    vector_store.EMBEDDINGS_FILE = store_dir / "embeddings.npy"
    vector_store.METADATA_FILE = store_dir / "metadata.json"
    document_registry.STORE_DIR = store_dir
    document_registry.REGISTRY_FILE = store_dir / "documents.json"


def _index_corpus(pdf_dir: Path) -> None:
    for path in build_corpus(pdf_dir):
        data = path.read_bytes()
        doc_id = hashlib.sha256(data).hexdigest()
        chunks = chunk_pages(load_pdf(path, doc_name=path.name), chunk_size=CHUNK_SIZE, overlap=OVERLAP)
        for chunk in chunks:
            chunk["text"] = sanitize_chunk_text(chunk["text"])
            chunk["doc_id"] = doc_id
        vector_store.add_chunks(chunks, embed_texts([c["text"] for c in chunks]))
        document_registry.add(doc_id, path.name, "public", path.name)


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, cwd=ROOT,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def run() -> dict:
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    answerable = [q for q in dataset["questions"] if q["evidence"]]

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _isolate_storage(tmp / "index")
        _index_corpus(tmp / "pdfs")
        _, all_chunks = vector_store._load()

        retrieve("warm-up query", top_k=10)  # loads the embedding model; not timed

        per_question = []
        latencies = []
        for q in answerable:
            start = time.perf_counter()
            results = retrieve(q["question"], top_k=10)
            latencies.append((time.perf_counter() - start) * 1000)
            per_question.append({
                "id": q["id"],
                "category": q["category"],
                "recall_at_5": recall_at_k(results, q["evidence"], 5),
                "reciprocal_rank_at_10": reciprocal_rank(results, q["evidence"], 10),
                "top_5": [f"{c['doc_name']} p{c['page']}" for c in results[:5]],
            })

    evidence_items = [ev for q in answerable for ev in q["evidence"]]
    intact = [ev for ev in evidence_items if evidence_intact(all_chunks, ev)]

    by_category = {}
    for row in per_question:
        by_category.setdefault(row["category"], []).append(row["recall_at_5"])

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "config": {
            "search_method": "semantic",
            "reranking": False,
            "embedding_model": MODEL_NAME,
            "chunking": f"fixed {CHUNK_SIZE} words / {OVERLAP} overlap",
        },
        "dataset_version": dataset["dataset_version"],
        "corpus_version": CORPUS_VERSION,
        "counts": {
            "questions_total": len(dataset["questions"]),
            "questions_scored": len(answerable),
            "chunks_indexed": len(all_chunks),
        },
        "metrics": {
            "recall_at_5": round(statistics.mean(r["recall_at_5"] for r in per_question), 3),
            "mrr_at_10": round(statistics.mean(r["reciprocal_rank_at_10"] for r in per_question), 3),
            "evidence_intactness": round(len(intact) / len(evidence_items), 3),
            "recall_at_5_by_category": {k: round(statistics.mean(v), 3) for k, v in by_category.items()},
            "retrieval_latency_ms": {
                "p50": round(percentile(latencies, 50), 1),
                "p95": round(percentile(latencies, 95), 1),
            },
        },
        "split_evidence": [f"{ev['doc']} p{ev['page']}: {ev['phrase']}" for ev in evidence_items if ev not in intact],
        "failed_questions": [r for r in per_question if r["recall_at_5"] < 1.0],
        "per_question": per_question,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--name", help="results file name (default: timestamp)")
    args = parser.parse_args()

    report = run()
    RESULTS_DIR.mkdir(exist_ok=True)
    name = args.name or report["timestamp"].replace(":", "-")
    out = RESULTS_DIR / f"{name}.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    m = report["metrics"]
    print(f"Recall@5 {m['recall_at_5']}  MRR@10 {m['mrr_at_10']}  "
          f"intactness {m['evidence_intactness']}  p95 {m['retrieval_latency_ms']['p95']} ms  "
          f"chunks {report['counts']['chunks_indexed']}")
    print(f"Failed questions: {[q['id'] for q in report['failed_questions']]}")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

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

from evaluation.corpus import build_corpus
from evaluation.retrieval_metrics import evidence_intact, percentile, recall_at_k, reciprocal_rank
from rag import document_registry, settings, vector_store
from rag.embeddings import MODEL_NAME, embed_texts, get_model
from rag.pdf_loader import load_pdf
from rag.retriever import retrieve
from rag.text_chunker import STRATEGIES, chunk_document
from security.prompt_guard import sanitize_chunk_text

ROOT = Path(__file__).parent
DATASET = ROOT / "benchmark_dataset.json"
RESULTS_DIR = ROOT / "results"

SWEEP_SIZES = [(150, 30), (300, 50), (500, 75)]


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


def _index_corpus(pdf_dir: Path, corpus_version: int) -> None:
    for path in build_corpus(pdf_dir, version=corpus_version):
        data = path.read_bytes()
        doc_id = hashlib.sha256(data).hexdigest()
        chunks = chunk_document(load_pdf(path, doc_name=path.name), settings.CHUNK_STRATEGY,
                                settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
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


def run(strategy: str = "fixed", size: int = 150, overlap: int = 30, corpus_version: int = 2) -> dict:
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    answerable = [q for q in dataset["questions"] if q["evidence"]]
    saved = (settings.CHUNK_STRATEGY, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    # The run's chunk settings, so the index manifest describes what was built.
    settings.CHUNK_STRATEGY, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP = strategy, size, overlap
    try:
        return _run(dataset, answerable, strategy, size, overlap, corpus_version)
    finally:
        settings.CHUNK_STRATEGY, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP = saved


def _run(dataset, answerable, strategy, size, overlap, corpus_version) -> dict:
    embed_texts(["warm-up"])  # load the model before timing indexing

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        _isolate_storage(tmp / "index")
        start = time.perf_counter()
        _index_corpus(tmp / "pdfs", corpus_version)
        index_seconds = time.perf_counter() - start
        storage_bytes = sum(p.stat().st_size for p in (tmp / "index").iterdir() if p.suffix in (".npy", ".json"))
        _, all_chunks = vector_store._load()

        tokenizer, limit = get_model().tokenizer, get_model().max_seq_length
        over_limit = sum(len(tokenizer(c["text"])["input_ids"]) > limit for c in all_chunks)

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
            "chunking": f"{strategy} {size} words / {overlap} overlap",
        },
        "dataset_version": dataset["dataset_version"],
        "corpus_version": str(corpus_version),
        "counts": {
            "questions_total": len(dataset["questions"]),
            "questions_scored": len(answerable),
            "chunks_indexed": len(all_chunks),
        },
        "indexing": {
            "seconds": round(index_seconds, 2),
            "storage_kb": round(storage_bytes / 1024, 1),
            "chunks_over_model_limit_pct": round(100 * over_limit / len(all_chunks), 1),
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


def sweep() -> dict:
    """Every strategy x size on corpus v2, plus fixed 150/30 on v1 to tie back to the Phase 1 baseline."""
    rows = [run("fixed", 150, 30, corpus_version=1)]
    rows += [run(st, s, o, corpus_version=2) for st in STRATEGIES for s, o in SWEEP_SIZES]
    keep = ("config", "corpus_version", "counts", "indexing", "metrics", "failed_questions")
    return {"timestamp": rows[0]["timestamp"], "git_commit": rows[0]["git_commit"],
            "runs": [{k: r[k] for k in keep} for r in rows]}


def _table(result: dict) -> str:
    lines = ["| Corpus | Chunking | Chunks | Over model limit | Index s | Storage KB | Recall@5 | MRR@10 | p50 / p95 ms |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in result["runs"]:
        m, i = r["metrics"], r["indexing"]
        lat = m["retrieval_latency_ms"]
        lines.append(f"| v{r['corpus_version']} | {r['config']['chunking']} | {r['counts']['chunks_indexed']} | "
                     f"{i['chunks_over_model_limit_pct']}% | {i['seconds']} | {i['storage_kb']} | "
                     f"{m['recall_at_5']} | {m['mrr_at_10']} | {lat['p50']} / {lat['p95']} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--name", help="results file name (default: timestamp)")
    parser.add_argument("--strategy", choices=STRATEGIES, default=settings.CHUNK_STRATEGY)
    parser.add_argument("--size", type=int, default=settings.CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=settings.CHUNK_OVERLAP)
    parser.add_argument("--corpus", type=int, choices=(1, 2), default=2)
    parser.add_argument("--sweep", action="store_true", help="run every strategy and size; writes phase2_sweep.json")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    if args.sweep:
        result = sweep()
        out = RESULTS_DIR / f"{args.name or 'phase2_sweep'}.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(_table(result))
        print(f"Saved {out}")
        return

    report = run(args.strategy, args.size, args.overlap, args.corpus)
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

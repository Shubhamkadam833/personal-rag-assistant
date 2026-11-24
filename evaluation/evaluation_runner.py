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
from evaluation.retrieval_metrics import (evidence_intact, ndcg_at_k, percentile, precision_at_k, recall_at_k,
                                          reciprocal_rank)
from rag import document_registry, settings, vector_store
from rag.embeddings import MODEL_NAME, embed_texts, get_model
from rag.pdf_loader import load_pdf
from rag.retriever import retrieve_with_info
from rag.text_chunker import STRATEGIES, chunk_document
from security.prompt_guard import sanitize_chunk_text

ROOT = Path(__file__).parent
DATASET = ROOT / "benchmark_dataset.json"
RESULTS_DIR = ROOT / "results"

SWEEP_SIZES = [(150, 30), (300, 50), (500, 75)]
POOL = 20  # candidates ranked per question (= settings.RERANK_CANDIDATES default)


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


def run(strategy: str = "fixed", size: int = 150, overlap: int = 30, corpus_version: int = 2,
        mode: str = "semantic", rerank: bool = False) -> dict:
    dataset = json.loads(DATASET.read_text(encoding="utf-8"))
    answerable = [q for q in dataset["questions"] if q["evidence"] and q.get("min_corpus", 1) <= corpus_version]
    saved = (settings.CHUNK_STRATEGY, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
    # The run's chunk settings, so the index manifest describes what was built.
    settings.CHUNK_STRATEGY, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP = strategy, size, overlap
    try:
        return _run(dataset, answerable, strategy, size, overlap, corpus_version, mode, rerank)
    finally:
        settings.CHUNK_STRATEGY, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP = saved


def _run(dataset, answerable, strategy, size, overlap, corpus_version, mode, rerank) -> dict:
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

        # Loads the embedding (and reranker) model; not timed.
        retrieve_with_info("warm-up query", top_k=POOL, mode=mode, rerank=rerank, candidates=POOL)

        per_question = []
        latencies, rerank_ms, fallbacks = [], [], []
        for q in answerable:
            # The whole candidate pool is ranked, so reranking on/off see the
            # same POOL chunks and only their order differs.
            start = time.perf_counter()
            results, info = retrieve_with_info(q["question"], top_k=POOL, mode=mode, rerank=rerank, candidates=POOL)
            latencies.append((time.perf_counter() - start) * 1000)
            rerank_ms.append(info["rerank_ms"])
            if info["rerank_fallback"]:
                fallbacks.append(f"{q['id']}: {info['rerank_fallback']}")
            per_question.append({
                "id": q["id"],
                "category": q["category"],
                "recall_at_5": recall_at_k(results, q["evidence"], 5),
                "reciprocal_rank_at_10": reciprocal_rank(results, q["evidence"], 10),
                "precision_at_5": precision_at_k(results, q["evidence"], 5),
                "ndcg_at_5": ndcg_at_k(results, q["evidence"], 5),
                "pool_recall_at_20": recall_at_k(results, q["evidence"], POOL),
                # Rough: ~4 characters per token for English text.
                "est_top5_tokens": sum(len(c["text"]) for c in results[:5]) // 4,
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
            "search_method": mode,
            "reranking": rerank,
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
            "precision_at_5": round(statistics.mean(r["precision_at_5"] for r in per_question), 3),
            "ndcg_at_5": round(statistics.mean(r["ndcg_at_5"] for r in per_question), 3),
            "pool_recall_at_20": round(statistics.mean(r["pool_recall_at_20"] for r in per_question), 3),
            "est_top5_tokens_mean": round(statistics.mean(r["est_top5_tokens"] for r in per_question)),
            "recall_at_5_by_category": {k: round(statistics.mean(v), 3) for k, v in by_category.items()},
            "rerank_latency_ms": ({"p50": round(percentile(rerank_ms, 50), 1), "p95": round(percentile(rerank_ms, 95), 1)}
                                  if rerank else None),
            "rerank_fallbacks": fallbacks,
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


def compare_modes(corpus_version: int = 2) -> dict:
    """Every search mode on the same corpus, chunking and questions (Phase 3)."""
    rows = [run(settings.CHUNK_STRATEGY, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP, corpus_version, mode)
            for mode in settings.SEARCH_MODES]
    keep = ("config", "corpus_version", "counts", "indexing", "metrics", "failed_questions", "per_question")
    return {"timestamp": rows[0]["timestamp"], "git_commit": rows[0]["git_commit"],
            "runs": [{k: r[k] for k in keep} for r in rows]}


def compare_rerank(corpus_version: int = 3) -> dict:
    """
    Phase 4: every search mode with reranking off and on, same corpus,
    questions and candidate pool. Plus, for hybrid (the default), which
    questions reranking improved, worsened or left unchanged.
    """
    rows = [run(settings.CHUNK_STRATEGY, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP, corpus_version, mode, rerank)
            for mode in settings.SEARCH_MODES for rerank in (False, True)]
    keep = ("config", "corpus_version", "counts", "metrics", "failed_questions", "per_question")
    hybrid_off, hybrid_on = (r for r in rows if r["config"]["search_method"] == "hybrid")
    return {"timestamp": rows[0]["timestamp"], "git_commit": rows[0]["git_commit"],
            "runs": [{k: r[k] for k in keep} for r in rows],
            "hybrid_changes": question_changes(hybrid_off["per_question"], hybrid_on["per_question"])}


def question_changes(before: list[dict], after: list[dict]) -> dict:
    """Per question: did nDCG@5 (then MRR@10) go up, down, or stay the same?"""
    out = {"improved": [], "worse": [], "unchanged": []}
    for b, a in zip(before, after):
        old = (round(b["ndcg_at_5"], 4), round(b["reciprocal_rank_at_10"], 4))
        new = (round(a["ndcg_at_5"], 4), round(a["reciprocal_rank_at_10"], 4))
        bucket = "improved" if new > old else "worse" if new < old else "unchanged"
        out[bucket].append({"id": b["id"], "category": b["category"], "ndcg_at_5": [old[0], new[0]],
                            "mrr_at_10": [old[1], new[1]], "top_5_before": b["top_5"], "top_5_after": a["top_5"]})
    return out


def _rerank_table(result: dict) -> str:
    lines = ["| Search | Rerank | Recall@5 | Precision@5 | nDCG@5 | MRR@10 | Pool recall@20 | Search p50/p95 ms | Rerank p50/p95 ms |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in result["runs"]:
        m = r["metrics"]
        lat, rr = m["retrieval_latency_ms"], m["rerank_latency_ms"]
        rerank_cell = f"{rr['p50']} / {rr['p95']}" if rr else "-"
        lines.append(f"| {r['config']['search_method']} | {'on' if r['config']['reranking'] else 'off'} | "
                     f"{m['recall_at_5']} | {m['precision_at_5']} | {m['ndcg_at_5']} | {m['mrr_at_10']} | "
                     f"{m['pool_recall_at_20']} | {lat['p50']} / {lat['p95']} | {rerank_cell} |")
    ch = result["hybrid_changes"]
    lines.append(f"\nHybrid, reranking on vs off: {len(ch['improved'])} improved, {len(ch['worse'])} worse, "
                 f"{len(ch['unchanged'])} unchanged")
    return "\n".join(lines)


def _table(result: dict) -> str:
    lines = ["| Corpus | Search | Chunking | Chunks | Over model limit | Index s | Storage KB | Recall@5 | MRR@10 | p50 / p95 ms |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for r in result["runs"]:
        m, i = r["metrics"], r["indexing"]
        lat = m["retrieval_latency_ms"]
        lines.append(f"| v{r['corpus_version']} | {r['config']['search_method']} | {r['config']['chunking']} | {r['counts']['chunks_indexed']} | "
                     f"{i['chunks_over_model_limit_pct']}% | {i['seconds']} | {i['storage_kb']} | "
                     f"{m['recall_at_5']} | {m['mrr_at_10']} | {lat['p50']} / {lat['p95']} |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--name", help="results file name (default: timestamp)")
    parser.add_argument("--strategy", choices=STRATEGIES, default=settings.CHUNK_STRATEGY)
    parser.add_argument("--size", type=int, default=settings.CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=settings.CHUNK_OVERLAP)
    parser.add_argument("--corpus", type=int, choices=(1, 2, 3), default=2)
    parser.add_argument("--sweep", action="store_true", help="run every strategy and size; writes phase2_sweep.json")
    parser.add_argument("--modes", action="store_true", help="run every search mode; writes phase3_search_modes.json")
    parser.add_argument("--mode", choices=settings.SEARCH_MODES, default=settings.SEARCH_MODE)
    parser.add_argument("--rerank", action="store_true", help="rerank the candidate pool (single run)")
    parser.add_argument("--rerank-compare", action="store_true",
                        help="every mode with reranking off/on on corpus v3; writes phase4_rerank.json")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)
    if args.rerank_compare:
        result = compare_rerank()
        out = RESULTS_DIR / f"{args.name or 'phase4_rerank'}.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(_rerank_table(result))
        print(f"Saved {out}")
        return
    if args.sweep or args.modes:
        result = sweep() if args.sweep else compare_modes(args.corpus)
        out = RESULTS_DIR / f"{args.name or ('phase2_sweep' if args.sweep else 'phase3_search_modes')}.json"
        out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(_table(result))
        print(f"Saved {out}")
        return

    report = run(args.strategy, args.size, args.overlap, args.corpus, args.mode, args.rerank)
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

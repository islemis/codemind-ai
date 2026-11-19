
import argparse
import json
import math
import time
from functools import lru_cache
from pathlib import Path

from sentence_transformers import CrossEncoder

from backend.local_project_service import load_local_project
from backend.services.embedding_service import prepare_document
from backend.services.qdrant_service import (
    client,
    COLLECTION_NAME,
    search_components
)


# ==========================================
# CONFIGURATION
# ==========================================

K = 5
RERANK_CANDIDATES = 20

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

FOLDER = Path(__file__).resolve().parent
RESULTS_FOLDER = FOLDER / "results"

PROJECT_PATH = (
    "C:/Users/ASUS/Desktop/projects/"
    "AlloService/AlloService-main"
)


# ==========================================
# MODE D'EVALUATION
# ==========================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    choices=["bm25", "hybrid", "rerank"],
    required=True
)

MODE = parser.parse_args().mode


# ==========================================
# CHARGER QUESTIONS ET QRELS
# ==========================================

with open(
    FOLDER / "question_v2.jsonl",
    encoding="utf-8"
) as f:
    questions = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

with open(
    FOLDER / "qrels.json",
    encoding="utf-8"
) as f:
    qrels = json.load(f)

project = load_local_project(PROJECT_PATH)
project_path = str(Path(project["project_path"]).resolve())


# ==========================================
# VERIFIER LA COLLECTION ET LES QRELS
# ==========================================

if not client.collection_exists(COLLECTION_NAME):
    raise RuntimeError(
        f"Collection {COLLECTION_NAME} not found. "
        "Index the project first."
    )

if not questions:
    raise RuntimeError("No benchmark questions found.")

for q in questions:
    if not qrels.get(q["query_id"]):
        raise RuntimeError(
            f"Missing Qrels for {q['query_id']}"
        )

relevant_ids = {
    str(chunk_id)
    for q in questions
    for chunk_id in qrels[q["query_id"]]
}

points = client.retrieve(
    collection_name=COLLECTION_NAME,
    ids=sorted(relevant_ids),
    with_payload=True,
    with_vectors=False
)

found_ids = {str(point.id) for point in points}
missing_ids = relevant_ids - found_ids

if missing_ids:
    raise RuntimeError(
        f"{len(missing_ids)} Qrels IDs missing "
        "from the hybrid collection."
    )

for point in points:
    if (point.payload or {}).get("project_path") != project_path:
        raise RuntimeError(
            f"Project mismatch for chunk {point.id}"
        )

print("Qrels validation: PASS")


# ==========================================
# RERANKER
# ==========================================

@lru_cache(maxsize=1)
def get_reranker():
    return CrossEncoder(RERANK_MODEL)


def rerank_results(question, candidates, top_k):
    if not candidates:
        return []

    model = get_reranker()

    pairs = [
        [question, prepare_document(chunk)]
        for chunk in candidates
    ]

    scores = model.predict(
        pairs,
        batch_size=16,
        show_progress_bar=False
    )

    ranked = sorted(
        zip(scores, candidates),
        key=lambda item: float(item[0]),
        reverse=True
    )

    return [
        {
            **chunk,
            "rerank_score": float(score)
        }
        for score, chunk in ranked[:top_k]
    ]


# ==========================================
# RETRIEVAL
# ==========================================

def retrieve(question, mode):

    # ---------- BM25 ----------
    if mode == "bm25":
        return search_components(
            project_path=project_path,
            query=question,
            top_k=K,
            mode="bm25"
        )

    # ---------- HYBRID ----------
    if mode == "hybrid":
        return search_components(
            project_path=project_path,
            query=question,
            top_k=K,
            mode="hybrid"
        )

    # ---------- HYBRID + RERANKING ----------
    if mode == "rerank":

        # Step 1: Hybrid retrieves 20 candidates
        candidates = search_components(
            project_path=project_path,
            query=question,
            top_k=RERANK_CANDIDATES,
            mode="hybrid"
        )

        # Step 2: Cross-Encoder reranks candidates
        return rerank_results(
            question=question,
            candidates=candidates,
            top_k=K
        )

    raise ValueError(f"Unknown retrieval mode: {mode}")


# ==========================================
# WARMUP (EXCLUDED FROM LATENCY)
# ==========================================

print("\nInitializing retrieval models...")

retrieve(
    question="Where is authentication implemented?",
    mode=MODE
)


# ==========================================
# EVALUATION
# ==========================================

results = []

for q in questions:

    query_id = q["query_id"]

    relevant = {
        str(chunk_id)
        for chunk_id in qrels[query_id]
    }

    start_time = time.perf_counter()

    # Retrieve using selected mode
    hits = retrieve(
        question=q["question"],
        mode=MODE
    )

    latency = time.perf_counter() - start_time

    retrieved = [
        str(hit["chunk_id"])
        for hit in hits
    ]

    correct = set(retrieved) & relevant

    # Precision@5
    precision = len(correct) / K

    # Recall@5
    recall = len(correct) / len(relevant)

    # MRR@5
    mrr = 0.0

    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            mrr = 1 / rank
            break

    # nDCG@5
    dcg = sum(
        1 / math.log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved, start=1)
        if chunk_id in relevant
    )

    ideal_count = min(len(relevant), K)

    idcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(1, ideal_count + 1)
    )

    ndcg = dcg / idcg if idcg else 0.0

    # Hit Rate@5
    hit_rate = 1 if correct else 0

    results.append({
        "query_id": query_id,
        "retrieved_ids": retrieved,
        "precision": precision,
        "recall": recall,
        "mrr": mrr,
        "ndcg": ndcg,
        "hit_rate": hit_rate,
        "latency": latency
    })

    print(
        f"{query_id}: "
        f"Precision={precision:.2f} | "
        f"Recall={recall:.2f} | "
        f"MRR={mrr:.2f}"
    )


# ==========================================
# MOYENNES
# ==========================================

if not results:
    raise RuntimeError("No questions evaluated.")

metrics = {}

for metric in [
    "precision",
    "recall",
    "mrr",
    "ndcg",
    "hit_rate",
    "latency"
]:
    metrics[metric] = sum(
        r[metric] for r in results
    ) / len(results)


print(f"\n===== {MODE.upper()} BENCHMARK =====")
print("Collection:", COLLECTION_NAME)
print("Evaluated questions:", len(results))
print("Skipped questions:", len(questions) - len(results))

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")


# ==========================================
# SAUVEGARDER LES RESULTATS
# ==========================================

RESULTS_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

output = {
    "method": MODE,
    "collection": COLLECTION_NAME,
    "k": K,
    "rerank_candidates": (
        RERANK_CANDIDATES if MODE == "rerank" else None
    ),
    "reranker": (
        RERANK_MODEL if MODE == "rerank" else None
    ),
    "evaluated": len(results),
    "skipped": len(questions) - len(results),
    "metrics": metrics,
    "details": results
}

output_file = (
    RESULTS_FOLDER / f"results_{MODE}_v2.json"
)

with open(
    output_file,
    "w",
    encoding="utf-8"
) as f:
    json.dump(output, f, indent=2)

print("\nSaved:", output_file)

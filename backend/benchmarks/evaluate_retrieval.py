
import argparse
import json
import math
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer, CrossEncoder

from backend.local_project_service import load_local_project


# ======================================
# CONFIGURATION
# ======================================

COLLECTION = "codemind_code_jina"

K = 7
CANDIDATE_K = 20

JINA_MODEL = "jinaai/jina-embeddings-v2-base-code"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

FOLDER = Path(__file__).resolve().parent
RESULTS_FOLDER = FOLDER / "results"

PROJECT_PATH = (
    "C:/Users/ASUS/Desktop/projects/"
    "AlloService/AlloService-main"
)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    choices=["jina", "jina_rerank"],
    required=True
)

MODE = parser.parse_args().mode


# ======================================
# CHARGER QUESTIONS ET QRELS
# ======================================

with open(FOLDER / "question_v2.jsonl", encoding="utf-8") as f:
    questions = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

with open(FOLDER / "qrels.json", encoding="utf-8") as f:
    qrels = json.load(f)


# ======================================
# QDRANT
# ======================================

client = QdrantClient(url="http://localhost:6333")

project = load_local_project(PROJECT_PATH)

project_filter = Filter(
    must=[
        FieldCondition(
            key="project_path",
            match=MatchValue(value=project["project_path"])
        )
    ]
)

if not client.collection_exists(COLLECTION):
    raise RuntimeError(
        f"Collection {COLLECTION} does not exist."
    )


# ======================================
# INITIALISER LES MODELES
# ======================================

print("Loading Jina Dense model...")

jina_model = SentenceTransformer(
    JINA_MODEL,
    trust_remote_code=True
)

reranker = None

if MODE == "jina_rerank":
    print("Loading Cross-Encoder reranker...")
    reranker = CrossEncoder(RERANK_MODEL)


# Warmup hors chronometrage
jina_model.encode(
    ["warmup"],
    normalize_embeddings=True
)

if reranker is not None:
    reranker.predict(
        [["test question", "test document"]],
        show_progress_bar=False
    )


# ======================================
# RECHERCHE JINA + RERANKING
# ======================================

def retrieve(question):

    # 1. Embedding Jina de la question
    vector = jina_model.encode(
        [question],
        normalize_embeddings=True
    )[0].tolist()

    # 2. Recuperation des candidats
    limit = CANDIDATE_K if MODE == "jina_rerank" else K

    hits = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        query_filter=project_filter,
        limit=limit,
        with_payload=(MODE == "jina_rerank")
    ).points

    # Jina seul : Top 5 directement
    if MODE == "jina":
        return hits

    # 3. Preparer les paires question / code
    pairs = []

    for hit in hits:
        payload = hit.payload or {}

        code = payload.get("code", "")

        if not code:
            raise RuntimeError(
                f"Missing code in chunk {hit.id}"
            )

        document = (
            f"File: {payload.get('file', '')}\n"
            f"Symbol: {payload.get('name', '')}\n"
            f"Code:\n{code[:3000]}"
        )

        pairs.append([question, document])

    if not pairs:
        return []

    # 4. Scores du Cross-Encoder
    scores = reranker.predict(
        pairs,
        batch_size=16,
        show_progress_bar=False
    )

    # 5. Trier les 20 candidats selon leurs scores
    ranked = sorted(
        zip(hits, scores),
        key=lambda item: float(item[1]),
        reverse=True
    )

    # 6. Garder les 5 meilleurs
    return [
        hit
        for hit, score in ranked[:K]
    ]


# ======================================
# EVALUATION
# ======================================

results = []

for q in questions:

    query_id = q["query_id"]
    relevant = {
        str(chunk_id)
        for chunk_id in qrels.get(query_id, [])
    }

    if not relevant:
        print(f"{query_id}: skipped (no Qrels)")
        continue

    start_time = time.perf_counter()

    hits = retrieve(q["question"])

    latency = time.perf_counter() - start_time

    retrieved = [str(hit.id) for hit in hits]
    correct = set(retrieved) & relevant

    # Precision et Recall
    precision = len(correct) / K
    recall = len(correct) / len(relevant)

    # MRR
    mrr = 0.0

    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            mrr = 1 / rank
            break

    # nDCG
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

    # Hit Rate
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


# ======================================
# MOYENNES
# ======================================

if not results:
    raise ValueError("No questions evaluated.")

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
print("Collection:", COLLECTION)
print("Evaluated questions:", len(results))
print("Skipped questions:", len(questions) - len(results))

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")


# ======================================
# SAUVEGARDER LES RESULTATS
# ======================================

RESULTS_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

output = {
    "method": MODE,
    "collection": COLLECTION,
    "k": K,
    "candidate_k": CANDIDATE_K if MODE == "jina_rerank" else K,
    "reranker": RERANK_MODEL if MODE == "jina_rerank" else None,
    "evaluated": len(results),
    "skipped": len(questions) - len(results),
    "metrics": metrics,
    "details": results
}

output_file = RESULTS_FOLDER / f"results_{MODE}_v2.json"

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print("\nSaved:", output_file)

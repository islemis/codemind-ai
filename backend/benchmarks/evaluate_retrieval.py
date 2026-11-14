
import json
import math
import time
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from backend.local_project_service import load_local_project
from backend.services.embedding_service import generate_embeddings



# Configuration
COLLECTION = "codemind_code"
K = 5
FOLDER = Path(__file__).parent

PROJECT_PATH = (
    "C:/Users/ASUS/Desktop/projects/"
    "AlloService/AlloService-main"
)

# 1. Charger les questions et Qrels
with open(FOLDER / "questions.jsonl", encoding="utf-8") as f:
    questions = [
        json.loads(line)
        for line in f if line.strip()
    ]

with open(FOLDER / "qrels.json", encoding="utf-8") as f:
    qrels = json.load(f)

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

# 2. Initialiser le modèle (hors chronométrage)
generate_embeddings(["warmup"])

results = []

# 3. Évaluer chaque question
for q in questions:

    query_id = q["query_id"]
    relevant = set(qrels.get(query_id, []))

    if not relevant:
        print(f"{query_id}: skipped (no Qrels)")
        continue

    start_time = time.perf_counter()

    # Embedding de la question
    vector = generate_embeddings([q["question"]])[0]

    # Recherche dense dans Qdrant
    hits = client.query_points(
        collection_name=COLLECTION,
        query=vector,
        query_filter=project_filter,
        limit=K,
        with_payload=False
    ).points

    latency = time.perf_counter() - start_time

    # IDs des chunks retournés
    retrieved = [str(hit.id) for hit in hits]

    # Chunks pertinents retrouvés
    correct = set(retrieved) & relevant

    # Precision et Recall
    precision = len(correct) / K
    recall = len(correct) / len(relevant)

    # MRR : rang du premier chunk pertinent
    mrr = 0.0

    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            mrr = 1 / rank
            break

    # nDCG : qualité du classement
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


# 4. Moyennes sur toutes les questions évaluées
if not results:
    raise ValueError(
        "No questions evaluated. Check your qrels.json."
    )

metrics = {}

for metric in [
    "precision", "recall", "mrr",
    "ndcg", "hit_rate", "latency"
]:
    metrics[metric] = sum(
        r[metric] for r in results
    ) / len(results)

print("\n===== RETRIEVAL BENCHMARK =====")
print("Evaluated questions:", len(results))
print("Skipped questions:", len(questions) - len(results))

for name, value in metrics.items():
    print(f"{name}: {value:.4f}")

# 5. Sauvegarder les résultats
output = {
    "collection": COLLECTION,
    "k": K,
    "evaluated": len(results),
    "skipped": len(questions) - len(results),
    "metrics": metrics,
    "details": results
}

with open(FOLDER / "results"/ "results.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print("\nSaved benchmarks/results.json")

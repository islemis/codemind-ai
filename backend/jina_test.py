
import json
import math
import time
from pathlib import Path
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models
from backend.local_project_service import load_local_project
from backend.services.embedding_service import prepare_document

# Configuration
OLD = "codemind_code"
NEW = "codemind_code_jina"
K = 5

PROJECT_PATH = "C:/Users/ASUS/Desktop/projects/AlloService/AlloService-main"
FOLDER = Path(__file__).parent /  "benchmarks"

# 1. Charger Jina
model = SentenceTransformer(
    "jinaai/jina-embeddings-v2-base-code",
    trust_remote_code=True
)
model.max_seq_length = 1024

client = QdrantClient(url="http://localhost:6333")
project = load_local_project(PROJECT_PATH)

flt = models.Filter(
    must=[
        models.FieldCondition(
            key="project_path",
            match=models.MatchValue(value=project["project_path"])
        )
    ]
)

# 2. Lire les chunks existants de MiniLM
chunks = []
offset = None

while True:
    batch, offset = client.scroll(
        collection_name=OLD,
        scroll_filter=flt,
        limit=100,
        offset=offset,
        with_payload=True,
        with_vectors=False
    )
    chunks.extend(batch)

    if offset is None:
        break

if not chunks:
    raise ValueError("No AlloService chunks found in Qdrant.")

print("Original chunks:", len(chunks))

# 3. Créer la collection Jina
if not client.collection_exists(NEW):

    client.create_collection(
        collection_name=NEW,
        vectors_config=models.VectorParams(
            size=model.get_sentence_embedding_dimension(),
            distance=models.Distance.COSINE
        )
    )

    for i in range(0, len(chunks), 16):
        batch = chunks[i:i + 16]

        texts = [prepare_document(c.payload) for c in batch]

        vectors = model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=8
        )

        points = [
            models.PointStruct(
                id=c.id,
                vector=v.tolist(),
                payload=c.payload
            )
            for c, v in zip(batch, vectors)
        ]

        client.upsert(collection_name=NEW, points=points)

    print("Jina indexing completed.")

# Vérifier que l'index Jina est complet
count = client.count(
    collection_name=NEW,
    count_filter=flt,
    exact=True
).count

if count != len(chunks):
    raise ValueError(
        f"Incomplete Jina collection: {count}/{len(chunks)} chunks"
    )

# 4. Charger les questions et les Qrels
with open(FOLDER / "question_v2.jsonl", encoding="utf-8") as f:
    questions = [json.loads(line) for line in f if line.strip()]

with open(FOLDER / "qrels.json", encoding="utf-8") as f:
    qrels = json.load(f)

# 5. Benchmark
model.encode("warmup")
results = []

for q in questions:
    relevant = set(qrels.get(q["query_id"], []))

    if not relevant:
        print(q["query_id"], "skipped: no Qrels")
        continue

    start = time.perf_counter()

    vector = model.encode(
        q["question"],
        normalize_embeddings=True
    ).tolist()

    hits = client.query_points(
        collection_name=NEW,
        query=vector,
        query_filter=flt,
        limit=K,
        with_payload=False
    ).points

    latency = time.perf_counter() - start

    retrieved = [str(h.id) for h in hits]
    correct = set(retrieved) & relevant

    precision = len(correct) / K
    recall = len(correct) / len(relevant)

    mrr = next(
        (1 / rank for rank, cid in enumerate(retrieved, 1)
         if cid in relevant),
        0.0
    )

    dcg = sum(
        1 / math.log2(rank + 1)
        for rank, cid in enumerate(retrieved, 1)
        if cid in relevant
    )

    idcg = sum(
        1 / math.log2(rank + 1)
        for rank in range(1, min(len(relevant), K) + 1)
    )

    results.append({
        "query_id": q["query_id"],
        "precision": precision,
        "recall": recall,
        "mrr": mrr,
        "ndcg": dcg / idcg,
        "hit_rate": int(bool(correct)),
        "latency": latency
    })

if not results:
    raise ValueError("No questions evaluated.")

# 6. Résultats moyens
metrics = {
    key: sum(r[key] for r in results) / len(results)
    for key in ["precision", "recall", "mrr",
                "ndcg", "hit_rate", "latency"]
}

output = {
    "collection": NEW,
    "k": K,
    "evaluated": len(results),
    "skipped": len(questions) - len(results),
    "metrics": metrics,
    "details": results
}

with open(FOLDER / "results_jina.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2)

print("\n===== JINA BENCHMARK =====")
print("Questions evaluated:", len(results))

for key, value in metrics.items():
    print(f"{key}: {value:.4f}")

print("\nSaved results_jina.json")

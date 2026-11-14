
import json
import re
from pathlib import Path
from qdrant_client import QdrantClient

COLLECTION = "codemind_code"
FOLDER = Path(__file__).parent
N = 5  # 5 mots consécutifs en commun


def words(text):
    return re.sub(r"\s+", " ", text).strip().split()


def ngrams(text):
    w = words(text)
    return {" ".join(w[i:i + N]) for i in range(len(w) - N + 1)}


def is_relevant(passage, chunk):
    # Vérifier si le chunk contient le passage
    compact = lambda text: re.sub(r"\s+", "", text)

    return (
        compact(passage) in compact(chunk)
        or bool(ngrams(passage) & ngrams(chunk))
    )


# 1. Connexion Qdrant
client = QdrantClient("localhost", port=6333)

# 2. Charger les 70 questions
with open(FOLDER / "questions.jsonl", encoding="utf-8") as f:
    questions = [json.loads(line) for line in f if line.strip()]

# 3. Récupérer les chunks et leurs vrais IDs
chunks = []
offset = None

while True:
    batch, offset = client.scroll(
        collection_name=COLLECTION,
        limit=100,
        offset=offset,
        with_payload=True,
        with_vectors=False
    )

    chunks.extend(batch)

    if offset is None:
        break

# 4. Construire les Qrels
qrels = {}

for q in questions:
    relevant_ids = []

    for chunk in chunks:
        data = chunk.payload or {}

        # Vérifier le fichier source
        if data.get("file", "").replace("\\", "/") != q["source"]:
            continue

        # Vérifier les lignes correspondantes
        start = data.get("start_line")
        end = data.get("end_line")

        if start is not None and end is not None:
            if start > q["end_line"] or end < q["start_line"]:
                continue

        # Vérifier le passage dans le chunk
        if is_relevant(q["passage"], data.get("code", "")):
            relevant_ids.append(str(chunk.id))

    qrels[q["query_id"]] = relevant_ids

    print(q["query_id"], ":", len(relevant_ids), "chunks")

# 5. Sauvegarder les Qrels
with open(FOLDER / "qrels.json", "w", encoding="utf-8") as f:
    json.dump(qrels, f, indent=2)

print("Qrels saved successfully!")

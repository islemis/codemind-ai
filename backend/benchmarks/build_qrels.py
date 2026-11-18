
import json
from pathlib import Path
from qdrant_client import QdrantClient

COLLECTION = "codemind_code"
PROJECT_PATH = "C:/Users/ASUS/Desktop/projects/AlloService/AlloService-main"
FOLDER = Path(__file__).parent


def normalize(text):
    return "".join(str(text).split())


def is_relevant(evidence, payload):

    # Vérifier le fichier source
    file = str(payload.get("file", "")).replace("\\", "/")

    if file != evidence["source"]:
        return False

    # Vérifier les lignes
    start = payload.get("start_line")
    end = payload.get("end_line")

    if start is None or end is None:
        return False

    if start > evidence["end_line"] or end < evidence["start_line"]:
        return False

    # Vérifier que le chunk contient une partie du passage
    chunk_code = normalize(payload.get("code", ""))

    lines = [
        normalize(line)
        for line in evidence["passage"].splitlines()
        if len(normalize(line)) >= 12
    ]

    return any(line in chunk_code for line in lines)


# 1. Connexion Qdrant
client = QdrantClient(url="http://localhost:6333")

# 2. Charger les questions
with open(FOLDER / "question_v2.jsonl", encoding="utf-8") as f:
    questions = [json.loads(line) for line in f if line.strip()]

# 3. Récupérer tous les chunks Qdrant
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

# Garder uniquement les chunks d'AlloService
chunks = [
    chunk for chunk in chunks
    if str((chunk.payload or {}).get("project_path", "")).replace("\\", "/").lower().rstrip("/")
    == PROJECT_PATH.lower().rstrip("/")
]

if not chunks:
    raise ValueError("No chunks found for AlloService.")

# 4. Construire les Qrels
qrels = {}

for q in questions:

    relevant_ids = set()

    # Une question peut avoir plusieurs passages de référence
    for evidence in q["evidence"]:

        for chunk in chunks:
            if is_relevant(evidence, chunk.payload or {}):
                relevant_ids.add(str(chunk.id))

    qrels[q["query_id"]] = sorted(relevant_ids)

    print(
        f"{q['query_id']}: "
        f"{len(relevant_ids)} relevant chunks"
    )

# 5. Sauvegarder
with open(FOLDER / "qrels.json", "w", encoding="utf-8") as f:
    json.dump(qrels, f, indent=2)

print("\nQrels saved successfully!")

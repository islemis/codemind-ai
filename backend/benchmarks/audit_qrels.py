
import json
import csv
from pathlib import Path
from qdrant_client import QdrantClient

# Paths
FOLDER = Path(__file__).resolve().parent

QUESTIONS_FILE = FOLDER / "question_v2.jsonl"
QRELS_FILE = FOLDER / "qrels.json"
OUTPUT_FILE = FOLDER / "qrels_review.csv"

COLLECTION = "codemind_code"
PROJECT_PATH = "C:/Users/ASUS/Desktop/projects/AlloService/AlloService-main"

# 1. Load questions
with open(QUESTIONS_FILE, encoding="utf-8") as f:
    questions = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

# 2. Load Qrels
with open(QRELS_FILE, encoding="utf-8") as f:
    qrels = json.load(f)

# 3. Connect to Qdrant
client = QdrantClient(url="http://localhost:6333")

# 4. Retrieve actual chunks and their IDs
chunks = {}
offset = None

def normalize_path(path):
    return str(path).replace("\\", "/").lower().rstrip("/")

while True:
    batch, offset = client.scroll(
        collection_name=COLLECTION,
        limit=100,
        offset=offset,
        with_payload=True,
        with_vectors=False
    )

    for chunk in batch:
        payload = chunk.payload or {}

        if normalize_path(payload.get("project_path", "")) == normalize_path(PROJECT_PATH):
            chunks[str(chunk.id)] = payload

    if offset is None:
        break

print("Chunks retrieved:", len(chunks))

# 5. Export Qrels + real chunk contents
rows = []

for q in questions:
    query_id = q["query_id"]
    relevant_ids = qrels.get(query_id, [])

    evidences = q.get("evidence", [])

    expected_code = "\n\n".join(
        f"FILE: {e['source']}\n"
        f"LINES: {e['start_line']}-{e['end_line']}\n"
        f"{e['passage']}"
        for e in evidences
    )

    for chunk_id in relevant_ids:
        payload = chunks.get(str(chunk_id))

        rows.append({
            "query_id": query_id,
            "question": q["question"],
            "chunk_id": str(chunk_id),
            "chunk_file": payload.get("file", "") if payload else "NOT FOUND",
            "start_line": payload.get("start_line", "") if payload else "",
            "end_line": payload.get("end_line", "") if payload else "",
            "chunk_code": payload.get("code", "") if payload else "",
            "expected_evidence": expected_code,
            "manual_review": ""
        })

# 6. Save CSV
columns = [
    "query_id",
    "question",
    "chunk_id",
    "chunk_file",
    "start_line",
    "end_line",
    "chunk_code",
    "expected_evidence",
    "manual_review"
]

with open(OUTPUT_FILE, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)

print("Questions:", len(questions))
print("Qrels associations:", len(rows))
print("CSV generated:", OUTPUT_FILE)

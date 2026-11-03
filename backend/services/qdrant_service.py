
import uuid
from pathlib import Path
from qdrant_client import QdrantClient, models

from backend.services.embedding_service import (
    prepare_document,
    generate_embeddings
)

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "codemind_code"

client = QdrantClient(url=QDRANT_URL)


def index_components(project_path: str, components: list[dict]):
    if not components:
        raise ValueError("No components found.")

    # Create the collection if it does not exist
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            )
        )

    project = str(Path(project_path).resolve())

    # Generate embeddings
    documents = [prepare_document(c) for c in components]
    vectors = generate_embeddings(documents)

    points = []

    for component, vector in zip(components, vectors):
        symbol = (
            f"{project}:{component['file']}:"
            f"{component['name']}:{component['start_line']}"
        )

        points.append(
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, symbol)),
                vector=vector,
                payload={
                    **component,
                    "project_path": project
                }
            )
        )

    # Store vectors and metadata in Qdrant
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True
    )

    return {
        "collection": COLLECTION_NAME,
        "indexed_components": len(points)
    }


def search_components(project_path: str, query: str, top_k: int = 5):
    if not client.collection_exists(COLLECTION_NAME):
        raise ValueError("Project has not been indexed.")

    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    project = str(Path(project_path).resolve())
    vector = generate_embeddings([query])[0]

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="project_path",
                    match=models.MatchValue(value=project)
                )
            ]
        ),
        limit=top_k,
        with_payload=True
    ).points

    return [
        {
            "score": round(result.score, 4),
            **result.payload
        }
        for result in results
    ]

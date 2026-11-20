
import uuid
from functools import lru_cache
from pathlib import Path

from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer

from backend.services.embedding_service import prepare_document


# Configuration
QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "codemind_code_jina"
JINA_MODEL = "jinaai/jina-embeddings-v2-base-code"

client = QdrantClient(url=QDRANT_URL)


# Charger Jina une seule fois
@lru_cache(maxsize=1)
def get_jina_model():
    model = SentenceTransformer(
        JINA_MODEL,
        trust_remote_code=True
    )
    model.max_seq_length = 1024
    return model


def index_components(project_path: str, components: list[dict]):
    if not components:
        raise ValueError("No components found.")

    model = get_jina_model()
    dimension = model.get_sentence_embedding_dimension()

    # Créer la collection Jina si nécessaire
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=dimension,  # 768 pour Jina
                distance=models.Distance.COSINE
            )
        )
    else:
        # Vérifier la compatibilité des vecteurs
        config = client.get_collection(
            COLLECTION_NAME
        ).config.params.vectors

        if not isinstance(config, models.VectorParams) or config.size != dimension:
            raise ValueError(
                f"Collection incompatible avec Jina ({dimension} dimensions)."
            )

    project = str(Path(project_path).resolve())

    # Préparer les documents comme dans jina_test
    documents = [
        prepare_document(component)
        for component in components
    ]

    # Générer les embeddings Jina
    vectors = model.encode(
        documents,
        normalize_embeddings=True,
        batch_size=8
    )

    points = []

    for component, vector in zip(components, vectors):
        symbol = (
            f"{project}:{component['file']}:"
            f"{component['name']}:{component['start_line']}"
        )

        points.append(
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, symbol)),
                vector=vector.tolist(),
                payload={
                    **component,
                    "project_path": project
                }
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True
    )

    return {
        "collection": COLLECTION_NAME,
        "indexed_components": len(points)
    }


def search_components(project_path: str, query: str, top_k: int = 7):
    if not client.collection_exists(COLLECTION_NAME):
        raise ValueError("Project has not been indexed.")

    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    project = str(Path(project_path).resolve())

    # Même modèle et paramètres que le benchmark
    model = get_jina_model()

    vector = model.encode(
        query,
        normalize_embeddings=True
    ).tolist()

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

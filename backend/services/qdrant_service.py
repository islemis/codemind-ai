
import uuid
from pathlib import Path

from qdrant_client import QdrantClient, models

from backend.services.embedding_service import (
    prepare_document,
    generate_jina_embeddings,
    generate_bm25_embeddings,
    generate_bm25_query,
    get_jina_embedding_dimension
)


QDRANT_URL = "http://localhost:6333"

# New collection; old MiniLM and Jina collections remain intact
COLLECTION_NAME = "codemind_code_jina"

DENSE_VECTOR = "jina_dense"
SPARSE_VECTOR = "bm25"

client = QdrantClient(url=QDRANT_URL, timeout=90)


def to_sparse_vector(embedding):
    return models.SparseVector(
        indices=[int(x) for x in embedding.indices],
        values=[float(x) for x in embedding.values]
    )


def project_filter(project_path: str):
    return models.Filter(
        must=[
            models.FieldCondition(
                key="project_path",
                match=models.MatchValue(
                    value=project_path
                )
            )
        ]
    )


def ensure_collection():
    if client.collection_exists(COLLECTION_NAME):
        config = client.get_collection(
            COLLECTION_NAME
        ).config.params

        dense = config.vectors
        sparse = config.sparse_vectors or {}

        expected_size = get_jina_embedding_dimension()

        if (
            not isinstance(dense, dict)
            or DENSE_VECTOR not in dense
            or dense[DENSE_VECTOR].size != expected_size
            or SPARSE_VECTOR not in sparse
            or sparse[SPARSE_VECTOR].modifier != models.Modifier.IDF
        ):
            raise RuntimeError(
                "Existing collection has an incompatible schema."
            )

        return

    client.create_collection(
        collection_name=COLLECTION_NAME,

        vectors_config={
            DENSE_VECTOR: models.VectorParams(
                size=get_jina_embedding_dimension(),
                distance=models.Distance.COSINE
            )
        },

        sparse_vectors_config={
            SPARSE_VECTOR: models.SparseVectorParams(
                modifier=models.Modifier.IDF
            )
        }
    )

    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="project_path",
        field_schema=models.PayloadSchemaType.KEYWORD,
        wait=True
    )


# ==================================================
# INDEXATION
# ==================================================

def index_components(project_path: str, components: list[dict]):
    if not components:
        raise ValueError("No components found.")

    project = str(Path(project_path).resolve())

    # 1. Prepare documents exactly as before
    documents = [
        prepare_document(component)
        for component in components
    ]

    # 2. Generate new dense and sparse vectors
    print("Generating Jina dense embeddings...")
    dense_vectors = generate_jina_embeddings(documents)

    print("Generating BM25 sparse embeddings...")
    sparse_vectors = generate_bm25_embeddings(documents)

    if not (
        len(components)
        == len(dense_vectors)
        == len(sparse_vectors)
    ):
        raise RuntimeError("Embedding count mismatch.")

    # 3. Preserve existing deterministic IDs
    points = []

    for component, dense, sparse in zip(
        components,
        dense_vectors,
        sparse_vectors
    ):
        symbol = (
            f"{project}:{component['file']}:"
            f"{component['name']}:{component['start_line']}"
        )

        chunk_id = str(
            uuid.uuid5(uuid.NAMESPACE_URL, symbol)
        )

        points.append(
            models.PointStruct(
                id=chunk_id,

                vector={
                    DENSE_VECTOR: dense,
                    SPARSE_VECTOR: to_sparse_vector(sparse)
                },

                payload={
                    **component,
                    "project_path": project
                }
            )
        )

    if len({str(p.id) for p in points}) != len(points):
        raise RuntimeError("Duplicate chunk IDs detected.")

    # 4. Create or validate the hybrid collection
    ensure_collection()

    # 5. Save to Qdrant
    for start in range(0, len(points), 32):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[start:start + 32],
            wait=True
        )

    # 6. Remove obsolete points belonging to this project
    #    only after the new points are saved.
    current_ids = {str(p.id) for p in points}
    stale_ids = []
    offset = None

    while True:
        batch, offset = client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=project_filter(project),
            offset=offset,
            limit=100,
            with_payload=False,
            with_vectors=False
        )

        stale_ids.extend(
            point.id for point in batch
            if str(point.id) not in current_ids
        )

        if offset is None:
            break

    if stale_ids:
        client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=models.PointIdsList(
                points=stale_ids
            ),
            wait=True
        )

    return {
        "collection": COLLECTION_NAME,
        "indexed_components": len(points),
        "removed_stale_points": len(stale_ids),
        "dense_model": "jina",
        "sparse_model": "bm25"
    }


# ==================================================
# RETRIEVAL
# ==================================================

def search_components(
    project_path: str,
    query: str,
    top_k: int = 5,
    mode: str = "dense"
):
    if not client.collection_exists(COLLECTION_NAME):
        raise ValueError("Project has not been indexed.")

    if not query.strip():
        raise ValueError("Search query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be positive.")

    if mode not in {"dense", "bm25", "hybrid"}:
        raise ValueError(f"Unsupported retrieval mode: {mode}")

    project = str(Path(project_path).resolve())
    filters = project_filter(project)

    # ---------- DENSE ----------
    if mode == "dense":
        dense_query = generate_jina_embeddings([query])[0]

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=dense_query,
            using=DENSE_VECTOR,
            query_filter=filters,
            limit=top_k,
            with_payload=True
        ).points

    # ---------- SPARSE / BM25 ----------
    elif mode == "bm25":
        sparse_query = to_sparse_vector(
            generate_bm25_query(query)
        )

        results = client.query_points(
            collection_name=COLLECTION_NAME,
            query=sparse_query,
            using=SPARSE_VECTOR,
            query_filter=filters,
            limit=top_k,
            with_payload=True
        ).points

    # ---------- HYBRID / RRF ----------
    else:
        dense_query = generate_jina_embeddings([query])[0]

        sparse_query = to_sparse_vector(
            generate_bm25_query(query)
        )

        candidate_limit = max(20, top_k * 4)

        results = client.query_points(
            collection_name=COLLECTION_NAME,

            prefetch=[
                models.Prefetch(
                    query=dense_query,
                    using=DENSE_VECTOR,
                    filter=filters,
                    limit=candidate_limit
                ),
                models.Prefetch(
                    query=sparse_query,
                    using=SPARSE_VECTOR,
                    filter=filters,
                    limit=candidate_limit
                )
            ],

            query=models.FusionQuery(
                fusion=models.Fusion.RRF
            ),

            limit=top_k,
            with_payload=True
        ).points

    return [
        {
            **(result.payload or {}),
            "chunk_id": str(result.id),
            "score": round(float(result.score), 4)
        }
        for result in results
    ]

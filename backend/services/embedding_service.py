
from functools import lru_cache

from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding


JINA_MODEL_NAME = "jinaai/jina-embeddings-v2-base-code"
BM25_MODEL_NAME = "Qdrant/bm25"



@lru_cache(maxsize=1)
def get_jina_model():
    return SentenceTransformer(
        JINA_MODEL_NAME,
        trust_remote_code=True
    )


def generate_jina_embeddings(texts: list[str]):
    model = get_jina_model()

    vectors = model.encode(
        texts,
        batch_size=16,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    return vectors.tolist()


def get_jina_embedding_dimension():
    return get_jina_model().get_sentence_embedding_dimension()



@lru_cache(maxsize=1)
def get_bm25_model():
    return SparseTextEmbedding(
        model_name=BM25_MODEL_NAME,
        disable_stemmer=True
    )


def generate_bm25_embeddings(texts: list[str]):
    """BM25 document embeddings for indexing."""
    return list(
        get_bm25_model().embed(texts, batch_size=16)
    )
def prepare_document(component: dict):
    parent = component.get("parent_class")

    name = component.get("name") or "unknown"
    full_name = f"{parent}.{name}" if parent else name

    return (
        f"Language: {component['language']}\n"
        f"File: {component['file']}\n"
        f"Symbol: {full_name}\n"
        f"Type: {component['type']}\n"
        f"Code:\n{component['code'][:3000]}"
    )

def generate_bm25_query(query: str):
    """BM25 query embedding for retrieval."""
    return next(
        get_bm25_model().query_embed(query)
    )

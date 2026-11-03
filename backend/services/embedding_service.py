
from functools import lru_cache
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_model():
    return SentenceTransformer(MODEL_NAME)


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


def generate_embeddings(texts: list[str]):
    model = get_model()

    vectors = model.encode(
        texts,
        batch_size=16,
        normalize_embeddings=True
    )

    return vectors.tolist()


def get_embedding_dimension():
    return get_model().get_sentence_embedding_dimension()

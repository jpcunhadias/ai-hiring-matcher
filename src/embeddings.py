from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils import logger

# Small (~80MB), no external API key, runs on CPU — keeps the project runnable
# with zero external accounts.
MODEL_NAME = "all-MiniLM-L6-v2"


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    logger.info("Carregando modelo de embeddings: %s", MODEL_NAME)
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str]) -> np.ndarray:
    """Encodes a list of texts into L2-normalized embeddings, so cosine similarity
    between any two rows reduces to a plain dot product."""
    model = get_embedding_model()
    return np.asarray(model.encode(texts, show_progress_bar=False, normalize_embeddings=True))

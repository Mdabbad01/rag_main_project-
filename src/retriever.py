from src.config import TOP_K
from src.vector_store import load_vector_store


def retrieve_relevant_chunks(query: str, k: int = None):
    """
    Retrieve the top-k most relevant chunks for a user query.
    """
    if k is None:
        k = TOP_K

    vector_store = load_vector_store()

    results = vector_store.similarity_search_with_score(query, k=k)

    return results
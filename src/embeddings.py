from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embedding_function():
    """
    Returns a local embedding model for vector generation.
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model
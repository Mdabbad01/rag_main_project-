from langchain_chroma import Chroma
from src.embeddings import get_embedding_function


def build_vector_store(chunks, reset_db=True):
    """
    Build an in-memory Chroma vector store from chunks.
    No disk persistence -> avoids Hugging Face readonly DB issues.
    """
    embedding_function = get_embedding_function()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function
    )

    return vector_store


def load_vector_store():
    """
    Disk-based loading disabled for Hugging Face stability.
    We will use Streamlit session_state instead.
    """
    raise RuntimeError("Persistent vector store loading is disabled. Use session_state vector store.")
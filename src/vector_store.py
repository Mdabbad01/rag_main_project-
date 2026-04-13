import os
import shutil
from langchain_chroma import Chroma
from src.config import CHROMA_PERSIST_DIR
from src.embeddings import get_embedding_function


def build_vector_store(chunks, reset_db=True):
    """
    Build and persist a Chroma vector store from chunks.
    If reset_db=True, it deletes the old DB and rebuilds it fresh.
    """
    if reset_db and os.path.exists(CHROMA_PERSIST_DIR):
        shutil.rmtree(CHROMA_PERSIST_DIR)

    embedding_function = get_embedding_function()

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_function,
        persist_directory=CHROMA_PERSIST_DIR
    )

    return vector_store


def load_vector_store():
    """
    Load an existing persisted Chroma vector store.
    """
    embedding_function = get_embedding_function()

    vector_store = Chroma(
        persist_directory=CHROMA_PERSIST_DIR,
        embedding_function=embedding_function
    )

    return vector_store
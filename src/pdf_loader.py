import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from src.config import DOCS_PATH


def load_documents():
    """
    Load all .pdf and .txt documents from the DOCS_PATH folder.
    Returns a list of LangChain Document objects with metadata.
    """
    if not os.path.exists(DOCS_PATH):
        raise FileNotFoundError(f"Docs path does not exist: {DOCS_PATH}")

    files = [
        f for f in os.listdir(DOCS_PATH)
        if f.lower().endswith(".pdf") or f.lower().endswith(".txt")
    ]

    if not files:
        raise FileNotFoundError(f"No .pdf or .txt files found in: {DOCS_PATH}")

    all_documents = []

    for file_name in files:
        file_path = os.path.join(DOCS_PATH, file_name)

        if file_name.lower().endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            docs = loader.load()

            for doc in docs:
                doc.metadata["source_file"] = file_name
                doc.metadata["file_type"] = "pdf"

        elif file_name.lower().endswith(".txt"):
            loader = TextLoader(file_path, encoding="utf-8")
            docs = loader.load()

            for doc in docs:
                doc.metadata["source_file"] = file_name
                doc.metadata["file_type"] = "txt"
                # mimic page field for consistency
                doc.metadata["page"] = 0

        all_documents.extend(docs)

    return all_documents
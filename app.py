from fastapi import FastAPI, Query
from pydantic import BaseModel

from src.config import OLLAMA_MODEL, CHROMA_PERSIST_DIR, DOCS_PATH, TOP_K
from src.pdf_loader import load_documents
from src.chunker import split_documents_into_chunks
from src.vector_store import build_vector_store
from src.retriever import retrieve_relevant_chunks
from src.rag_pipeline import ask_rag

app = FastAPI(
    title="Local PDF RAG Assistant",
    version="1.0.0",
    description="A local RAG project using FastAPI, ChromaDB, SentenceTransformers, and Ollama"
)


class AskRequest(BaseModel):
    query: str


@app.get("/")
def root():
    return {
        "message": "Welcome to the Local PDF RAG Assistant",
        "model": OLLAMA_MODEL
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ollama_model": OLLAMA_MODEL,
        "chroma_persist_dir": CHROMA_PERSIST_DIR,
        "docs_path": DOCS_PATH,
        "top_k": TOP_K
    }


@app.get("/test-docs")
def test_docs():
    docs = load_documents()

    preview = []
    for doc in docs[:5]:
        preview.append({
            "source_file": doc.metadata.get("source_file"),
            "file_type": doc.metadata.get("file_type"),
            "page": doc.metadata.get("page"),
            "text_preview": doc.page_content[:300]
        })

    return {
        "total_documents_loaded": len(docs),
        "preview": preview
    }


@app.get("/test-chunks")
def test_chunks():
    docs = load_documents()
    chunks = split_documents_into_chunks(docs)

    preview = []
    for chunk in chunks[:5]:
        preview.append({
            "chunk_id": chunk.metadata.get("chunk_id"),
            "source_file": chunk.metadata.get("source_file"),
            "file_type": chunk.metadata.get("file_type"),
            "page": chunk.metadata.get("page"),
            "chunk_length": len(chunk.page_content),
            "text_preview": chunk.page_content[:250]
        })

    return {
        "total_documents_loaded": len(docs),
        "total_chunks_created": len(chunks),
        "preview": preview
    }


@app.post("/build-db")
def build_db():
    docs = load_documents()
    chunks = split_documents_into_chunks(docs)
    vector_store = build_vector_store(chunks, reset_db=True)

    collection_count = vector_store._collection.count()

    return {
        "message": "Vector database built successfully",
        "total_documents_loaded": len(docs),
        "total_chunks_created": len(chunks),
        "vectors_stored": collection_count,
        "persist_directory": CHROMA_PERSIST_DIR
    }


@app.get("/test-retrieval")
def test_retrieval(q: str = Query(..., description="The user query")):
    results = retrieve_relevant_chunks(q)

    preview = []
    for doc, score in results:
        preview.append({
            "score": score,
            "source_file": doc.metadata.get("source_file"),
            "file_type": doc.metadata.get("file_type"),
            "page": doc.metadata.get("page"),
            "chunk_id": doc.metadata.get("chunk_id"),
            "text_preview": doc.page_content[:300]
        })

    return {
        "query": q,
        "top_k": len(results),
        "results": preview
    }


@app.post("/ask")
def ask_question(request: AskRequest):
    return ask_rag(request.query)
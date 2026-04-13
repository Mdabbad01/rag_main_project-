from src.vector_store import load_vector_store
from src.llm import generate_response
from src.config import TOP_K


def format_context(retrieved_docs):
    """
    Build context string from retrieved docs.
    """
    context_parts = []

    for idx, doc in enumerate(retrieved_docs, start=1):
        source = doc.metadata.get("source_file", "unknown")
        chunk_id = doc.metadata.get("chunk_id", "N/A")
        content = doc.page_content.strip()

        context_parts.append(
            f"[Source {idx} | file={source} | chunk={chunk_id}]\n{content}"
        )

    return "\n\n".join(context_parts)


def build_rag_prompt(query: str, context: str) -> str:
    """
    Strict grounded RAG prompt.
    """
    return f"""
You are a helpful enterprise policy assistant.

Answer the user's question ONLY using the provided context below.
If the answer is not clearly present in the context, say:
"I could not find the answer in the provided documents."

Context:
{context}

User Question:
{query}

Answer:
""".strip()


def build_general_prompt(query: str) -> str:
    """
    General fallback prompt for local Ollama.
    """
    return f"""
You are a helpful AI assistant running locally on Ollama.

Answer the user's question clearly and naturally.
If the question is personal like "what is your name", say that you are a local hybrid RAG assistant powered by Ollama.

User Question:
{query}

Answer:
""".strip()


def ask_rag(query: str, strict_mode: bool = False, score_threshold: float = 1.10):
    """
    Hybrid RAG pipeline.

    strict_mode=True:
        - only answer from retrieved documents
        - if retrieval is weak, refuse

    strict_mode=False:
        - try RAG first
        - if retrieval is weak, fallback to general LLM answer

    score_threshold:
        lower distance score = better match
        if best score <= threshold => good retrieval
        if best score > threshold => weak retrieval
    """
    vector_store = load_vector_store()

    # Retrieve with scores
    results = vector_store.similarity_search_with_score(query, k=TOP_K)

    if not results:
        if strict_mode:
            return {
                "query": query,
                "mode": "strict_rag",
                "answer": "I could not find the answer in the provided documents.",
                "sources": [],
                "used_fallback": False,
                "retrieval_status": "no_results"
            }
        else:
            fallback_prompt = build_general_prompt(query)
            fallback_answer = generate_response(fallback_prompt)

            return {
                "query": query,
                "mode": "hybrid_fallback",
                "answer": fallback_answer,
                "sources": [],
                "used_fallback": True,
                "retrieval_status": "no_results"
            }

    # Split docs and scores
    retrieved_docs = [doc for doc, score in results]
    source_entries = []

    for doc, score in results:
        source_entries.append({
            "source_file": doc.metadata.get("source_file", "unknown"),
            "file_type": doc.metadata.get("file_type", "unknown"),
            "page": doc.metadata.get("page", 0),
            "chunk_id": doc.metadata.get("chunk_id", "N/A"),
            "score": float(score),
            "content_preview": doc.page_content[:300]
        })

    best_score = float(results[0][1])

    # If retrieval is strong enough -> use RAG
    if best_score <= score_threshold:
        context = format_context(retrieved_docs)
        rag_prompt = build_rag_prompt(query, context)
        rag_answer = generate_response(rag_prompt)

        return {
            "query": query,
            "mode": "rag",
            "answer": rag_answer,
            "sources": source_entries,
            "used_fallback": False,
            "retrieval_status": "relevant_context_found",
            "best_score": best_score
        }

    # Retrieval weak
    if strict_mode:
        return {
            "query": query,
            "mode": "strict_rag",
            "answer": "I could not find the answer in the provided documents.",
            "sources": source_entries,
            "used_fallback": False,
            "retrieval_status": "weak_retrieval",
            "best_score": best_score
        }

    # Hybrid fallback to general LLM
    fallback_prompt = build_general_prompt(query)
    fallback_answer = generate_response(fallback_prompt)

    return {
        "query": query,
        "mode": "hybrid_fallback",
        "answer": fallback_answer,
        "sources": source_entries,
        "used_fallback": True,
        "retrieval_status": "weak_retrieval",
        "best_score": best_score
    }
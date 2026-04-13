
---


# Hybrid RAG Assistant

> A portfolio-ready **Hybrid Retrieval-Augmented Generation (RAG) Assistant** built with **Streamlit**, **FastAPI**, **ChromaDB**, **Sentence Transformers**, and a **pluggable LLM backend**.

This project supports two intelligent response modes:

- **Strict RAG Mode** → answers only from retrieved document context
- **Hybrid Mode** → uses document retrieval first, then falls back to general LLM knowledge when retrieval is weak

It is designed to demonstrate practical **GenAI engineering** concepts such as:

- document ingestion
- chunking
- embeddings
- vector search
- grounded answer generation
- retrieval confidence handling
- hybrid fallback logic
- local-to-cloud LLM migration

---

## ✨ Features

- Load `.txt` and `.pdf` documents from a local knowledge base
- Split documents into semantic chunks
- Generate embeddings using **Sentence Transformers**
- Store vectors in **ChromaDB**
- Retrieve top-k relevant chunks for a user query
- **Strict RAG Mode** for document-grounded answers only
- **Hybrid Mode** for RAG + general LLM fallback
- Interactive **Streamlit UI**
- Optional **FastAPI backend**
- Swappable LLM backend:
  - **Ollama** for local development
  - **Google Gemini API** for deployment-ready usage

---

## 🧠 Core Idea

Most beginner RAG projects stop at:

**Load documents → embed → retrieve → ask LLM**

This project goes one step further by adding:

- **retrieval score awareness**
- **strict vs hybrid response modes**
- **source transparency**
- **modular LLM backend design**

This makes it closer to how **real-world AI assistants** are designed.

---

## 🏗️ Architecture

```text
User Query
   ↓
Vector Retrieval (ChromaDB)
   ↓
Top-K Relevant Chunks
   ↓
Relevance Score Check
   ├── Strong retrieval  → Document-grounded RAG answer
   └── Weak retrieval    → Hybrid fallback to general LLM
---



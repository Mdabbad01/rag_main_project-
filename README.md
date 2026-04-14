
---
title: Enterprise Policy RAG Assistant
emoji: 🤖
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Enterprise Policy RAG Assistant

A portfolio-ready **Hybrid RAG (Retrieval-Augmented Generation)** project built with:

- **Streamlit** for the UI
- **FastAPI** for backend API endpoints
- **ChromaDB** as the vector database
- **Sentence Transformers** for embeddings
- **Groq API** for LLM inference
- **Docker** for deployment on Hugging Face Spaces

## Features

- Upload-free local knowledge base from `data/docs`
- Chunking + embedding + vector storage pipeline
- Hybrid retrieval mode:
  - **RAG Mode** for grounded document answers
  - **Fallback LLM Mode** for general questions
- Source chunk tracing
- Clean Streamlit UI
- Deployable on Hugging Face Spaces

## Project Structure

```bash
.
├── app.py                 # FastAPI backend
├── streamlit_app.py       # Streamlit frontend
├── Dockerfile             # Hugging Face Docker deployment
├── requirements.txt
├── README.md
├── data/
│   └── docs/
│       ├── employee_handbook.txt
│       ├── faq.txt
│       └── leave_policy.txt
└── src/
    ├── config.py
    ├── pdf_loader.py
    ├── chunker.py
    ├── embeddings.py
    ├── vector_store.py
    ├── retriever.py
    ├── llm.py
    └── rag_pipeline.py
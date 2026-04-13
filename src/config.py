import os
from dotenv import load_dotenv

load_dotenv()

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "chroma_db")
DOCS_PATH = os.getenv("DOCS_PATH", "data/docs")
TOP_K = int(os.getenv("TOP_K", 4))
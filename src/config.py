import os
from dotenv import load_dotenv

load_dotenv()

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "/tmp/chroma_db")
DOCS_PATH = os.getenv("DOCS_PATH", "data/docs")
TOP_K = int(os.getenv("TOP_K", 4))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
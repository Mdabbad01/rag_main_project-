from langchain_ollama import OllamaLLM
from src.config import OLLAMA_MODEL


def get_llm():
    """
    Return the local Ollama LLM instance.
    """
    return OllamaLLM(model=OLLAMA_MODEL)


def generate_response(prompt: str) -> str:
    """
    Generate a response from the local Ollama model.
    """
    llm = get_llm()
    response = llm.invoke(prompt)
    return response.strip() if isinstance(response, str) else str(response).strip()
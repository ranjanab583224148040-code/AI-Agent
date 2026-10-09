"""
Configuration module for Gemini Agent & RAG Pipeline.
Loads settings from environment variables and .env file.
"""

import os
from pathlib import Path

# Try loading from .env if python-dotenv is installed
try:
    from dotenv import load_dotenv
    # Look for .env in current file's directory
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()
except ImportError:
    pass

# Base project directory
BASE_DIR = Path(__file__).parent.resolve()

# Gemini API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Gemini LLM and Embedding Models
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite").strip()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001").strip()

# Paths
DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", str(BASE_DIR / "documents"))).resolve()
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db"))).resolve()

# Chunking Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# Retrieval Configuration
TOP_K = int(os.getenv("TOP_K", "4"))


def get_api_key() -> str:
    """Retrieve and validate the Gemini API key."""
    api_key = GEMINI_API_KEY or os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key or api_key == "YOUR_GEMINI_API_KEY_HERE":
        raise ValueError(
            "GEMINI_API_KEY is not set or still has the placeholder value.\n"
            "Please set your API key in the .env file or export it as an environment variable."
        )
    return api_key

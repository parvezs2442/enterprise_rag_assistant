import os
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Base backend directory: backend/
BACKEND_DIR = Path(__file__).resolve().parent.parent.parent

# Load .env from cwd or backend directory
env_path = BACKEND_DIR / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

class Settings(BaseModel):
    # API Keys & Providers
    google_api_key: str = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    gemini_chat_model: str = Field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest"))
    gemini_embed_model: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001"))

    # Storage Paths (auto-resolves correctly regardless of execution CWD)
    data_path: str = Field(default_factory=lambda: str(BACKEND_DIR / "data"))
    qdrant_path: str = Field(default_factory=lambda: str(BACKEND_DIR / "qdrant_db"))
    collection_name: str = "enterprise_rag_notes"

    # Chunking Hyperparameters
    chunk_size: int = 500
    chunk_overlap: int = 100

    # Retrieval Hyperparameters
    dense_k: int = 8
    bm25_k: int = 8
    rrf_k: int = 60
    rerank_top_n: int = 4

    # Index Management
    rebuild_index: bool = False

    def validate_keys(self):
        if not self.google_api_key:
            raise ValueError("ERROR: GOOGLE_API_KEY .env file mein nahi mila! Kripya .env file check karein.")

settings = Settings()
settings.validate_keys()

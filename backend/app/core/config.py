import os
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseModel):
    # API Keys & Providers
    google_api_key: str = Field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    gemini_chat_model: str = Field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-flash-lite-latest"))
    gemini_embed_model: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-001"))

    # Storage Paths
    data_path: str = "data"
    qdrant_path: str = "./qdrant_db"
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

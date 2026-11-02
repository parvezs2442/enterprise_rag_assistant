import os
from pathlib import Path
from dotenv import load_dotenv

# Base Directory (backend/)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# .env file load karna
load_dotenv(dotenv_path=BASE_DIR / ".env")


class Settings:
    PROJECT_NAME: str = "Enterprise RAG Assistant"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"

    # Gemini LLM Config
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "gemini-3.5-flash-lite")

    # Data directory (documents store karne ke liye)
    DATA_DIR: Path = BASE_DIR / "data"


settings = Settings()
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)

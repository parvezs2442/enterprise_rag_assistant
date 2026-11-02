from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from app.core.config import settings


class DocumentService:
    def load_pdf(self, file_path: str = None) -> List[Document]:
        """PDF file load karke Document objects list return karta hai."""
        if file_path is None:
            # First check in data/ directory
            in_data = settings.DATA_DIR / "notes.pdf"
            # Fallback check in loaders/ directory
            in_loaders = settings.BASE_DIR / "app" / "loaders" / "notes.pdf"

            if in_data.exists():
                path_to_load = in_data
            elif in_loaders.exists():
                path_to_load = in_loaders
            else:
                path_to_load = in_data
        else:
            path_to_load = Path(file_path)

        if not path_to_load.exists():
            raise FileNotFoundError(f"PDF file nahi mili: {path_to_load}")

        loader = PyPDFLoader(str(path_to_load))
        return loader.load()


document_service = DocumentService()

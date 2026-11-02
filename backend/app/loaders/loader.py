from pathlib import Path
from typing import List
from langchain_core.documents import Document
from app.services.document_service import document_service


def load_pdf(pdf_path: str = None) -> List[Document]:
    """
    Backward-compatible helper function to load PDF using document_service.
    """
    documents = document_service.load_pdf(pdf_path)
    print(f"PDF Loaded Successfully!")
    print(f"Total Pages: {len(documents)}")
    return documents


if __name__ == "__main__":
    docs = load_pdf()
    if docs:
        print("\n--- First Page Preview ---")
        print(docs[0].page_content[:300])
        print("\n--- Metadata ---")
        print(docs[0].metadata)

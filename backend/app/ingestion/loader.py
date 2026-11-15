import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

def load_pdf_documents(data_dir: str) -> List[Document]:
    """Folder ke saare PDFs ko load karke source metadata standardize karta hai."""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        raise FileNotFoundError(f"'{data_dir}' folder nahi mila! Kripya PDF documents dalein.")

    documents: List[Document] = []
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith(".pdf")]

    if not pdf_files:
        raise ValueError(f"'{data_dir}' folder mein koi PDF documents nahi mile!")

    for filename in pdf_files:
        file_path = os.path.join(data_dir, filename)
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        for page in pages:
            page.metadata["source"] = filename
        documents.extend(pages)
        print(f" -> Loaded PDF: {filename} ({len(pages)} pages)")

    return documents

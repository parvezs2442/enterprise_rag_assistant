
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def load_pdf():

    # Get PDF path (same directory as this file)
    pdf_path = Path(__file__).parent / "notes.pdf"

    # Load PDF
    loader = PyPDFLoader(str(pdf_path))

    # Extract pages
    documents = loader.load()

    print(f"PDF Loaded Successfully!")
    print(f"Total Pages: {len(documents)}")

    return documents

import shutil
from fastapi import APIRouter, HTTPException, UploadFile, File
from app.schemas.document import DocumentLoadResponse, DocumentPage
from app.services.document_service import document_service
from app.core.config import settings

router = APIRouter()


@router.get("/load-default", response_model=DocumentLoadResponse, summary="Load default notes.pdf")
def load_default_notes():
    """
    Default notes.pdf ko load karke pages aur metadata return karta hai.
    """
    try:
        docs = document_service.load_pdf()
        pages = [
            DocumentPage(
                page_number=i + 1,
                content=doc.page_content,
                metadata=doc.metadata
            )
            for i, doc in enumerate(docs)
        ]
        return DocumentLoadResponse(
            filename="notes.pdf",
            total_pages=len(pages),
            pages=pages
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Document load karne me error: {str(e)}"
        )


@router.post("/upload", response_model=DocumentLoadResponse, summary="Upload and load a new PDF")
def upload_pdf(file: UploadFile = File(...)):
    """
    Naya PDF file upload karke parse aur load karta hai.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Sirf PDF files allowed hain."
        )

    save_path = settings.DATA_DIR / file.filename
    try:
        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        docs = document_service.load_pdf(str(save_path))
        pages = [
            DocumentPage(
                page_number=i + 1,
                content=doc.page_content,
                metadata=doc.metadata
            )
            for i, doc in enumerate(docs)
        ]
        return DocumentLoadResponse(
            filename=file.filename,
            total_pages=len(pages),
            pages=pages
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Upload & load karne me error: {str(e)}"
        )

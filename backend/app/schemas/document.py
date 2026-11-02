from typing import List, Dict, Any
from pydantic import BaseModel


class DocumentPage(BaseModel):
    page_number: int
    content: str
    metadata: Dict[str, Any]


class DocumentLoadResponse(BaseModel):
    filename: str
    total_pages: int
    pages: List[DocumentPage]

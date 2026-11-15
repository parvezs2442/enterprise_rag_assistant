import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from app.core.config import settings

def init_bm25(chunks: List[Document], k: int = settings.bm25_k) -> BM25Retriever:
    """BM25 keyword search index initialize karta hai."""
    retriever = BM25Retriever.from_documents(chunks)
    retriever.k = k
    return retriever

def bm25_search(
    retriever: BM25Retriever,
    query: str,
    k: int = settings.bm25_k,
    source_filter: Optional[str] = None
) -> List[Document]:
    """BM25 token-frequency lexical matching execute karta hai."""
    retriever.k = k * 2 if source_filter else k
    results = retriever.invoke(query)
    if source_filter:
        results = [
            doc for doc in results
            if os.path.basename(doc.metadata.get("source", "")) == source_filter
        ]
    return results[:k]

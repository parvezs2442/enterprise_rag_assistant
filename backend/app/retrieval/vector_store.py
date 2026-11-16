import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from app.core.config import settings
from app.embeddings.gemini import GeminiEmbeddingsAdapter

def init_vector_store(
    chunks: List[Document],
    embeddings: GeminiEmbeddingsAdapter,
    rebuild: bool = False
) -> QdrantVectorStore:
    """Local Qdrant vector database load ya create karta hai."""
    client = QdrantClient(path=settings.qdrant_path)
    exists = client.collection_exists(settings.collection_name)

    if rebuild and exists:
        print(f" -> Deleting existing collection '{settings.collection_name}' for rebuild...")
        client.delete_collection(settings.collection_name)
        exists = False
    client.close()

    if not exists:
        print(f" -> Indexing {len(chunks)} chunks in Qdrant ({settings.qdrant_path})...")
        vectorstore = QdrantVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            path=settings.qdrant_path,
            collection_name=settings.collection_name,
            validate_collection_config=False
        )
        print(" -> Qdrant collection indexed successfully!")
    else:
        print(f" -> Loading existing Qdrant collection '{settings.collection_name}'...")
        vectorstore = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            path=settings.qdrant_path,
            collection_name=settings.collection_name,
            validate_collection_config=False
        )
        print(" -> Qdrant collection loaded.")

    return vectorstore

def dense_search(
    vectorstore: QdrantVectorStore,
    query: str,
    k: int = settings.dense_k,
    source_filter: Optional[str] = None
) -> List[Document]:
    """Semantic vector search run karta hai with optional metadata filtering."""
    q_filter = None
    if source_filter:
        q_filter = Filter(
            must=[
                FieldCondition(
                    key="metadata.source",
                    match=MatchValue(value=source_filter)
                )
            ]
        )
    return vectorstore.similarity_search(query=query, k=k, filter=q_filter)

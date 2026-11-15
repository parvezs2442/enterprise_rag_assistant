from typing import List
from langchain_core.documents import Document
from flashrank import Ranker, RerankRequest
from app.core.config import settings

class CrossEncoderReranker:
    """FlashRank TinyBERT Cross-Encoder Candidate Reranker."""
    def __init__(self):
        self.ranker = Ranker()

    def rerank(
        self,
        query: str,
        documents: List[Document],
        top_n: int = settings.rerank_top_n
    ) -> List[Document]:
        if not documents:
            return []

        passages = [
            {"id": i, "text": doc.page_content, "meta": doc.metadata}
            for i, doc in enumerate(documents)
        ]

        request = RerankRequest(query=query, passages=passages)
        ranked = self.ranker.rerank(request)

        reranked_docs = []
        for p in ranked[:top_n]:
            score = round(float(p.get("score", 0.0)), 4)
            doc = Document(
                page_content=p["text"],
                metadata={**p["meta"], "rerank_score": score}
            )
            reranked_docs.append(doc)

        return reranked_docs

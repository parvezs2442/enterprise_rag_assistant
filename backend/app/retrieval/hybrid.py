from collections import defaultdict
from typing import List, Dict, Tuple
from langchain_core.documents import Document
from app.core.config import settings

def reciprocal_rank_fusion(
    result_lists: List[List[Document]],
    k: int = settings.rrf_k
) -> List[Document]:
    """
    Reciprocal Rank Fusion (RRF) algorithm:
    Dense aur Sparse rank positions ko merge karta hai:
    Score(d) = sum(1 / (k + rank + 1))
    """
    scores: Dict[Tuple, float] = defaultdict(float)
    doc_map: Dict[Tuple, Document] = {}

    for doc_list in result_lists:
        for rank, doc in enumerate(doc_list):
            doc_key = (
                doc.metadata.get("source", ""),
                doc.metadata.get("page", 0),
                doc.metadata.get("chunk_id", 0),
                doc.page_content.strip()
            )
            scores[doc_key] += 1.0 / (k + rank + 1)
            doc_map[doc_key] = doc

    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [doc_map[key] for key in sorted_keys]

import re
from typing import List, Dict, Any
from langchain_core.documents import Document

def verify_citations(answer: str, documents: List[Document]) -> Dict[str, Any]:
    """
    Phase 9 Citation Verification Engine:
    - Answer se [DOC-X] tags ko extract karta hai.
    - Check karta hai ki kya har cited ID real retrieved documents se match karta hai.
    - Hallucinated / Fake citations ko flag karta hai.
    """
    total_docs = len(documents)
    valid_ids = {f"DOC-{i}" for i in range(1, total_docs + 1)}

    found_tags = set(re.findall(r"\[(DOC-\d+)\]", answer))
    valid_citations = found_tags.intersection(valid_ids)
    hallucinated_citations = found_tags.difference(valid_ids)

    id_to_metadata = {}
    for i, doc in enumerate(documents, start=1):
        id_to_metadata[f"DOC-{i}"] = {
            "source": doc.metadata.get("source", "Unknown"),
            "page": doc.metadata.get("page", 0) + 1,
            "score": doc.metadata.get("rerank_score", "N/A")
        }

    precision = len(valid_citations) / len(found_tags) if found_tags else 0.0

    return {
        "cited_tags": sorted(list(found_tags)),
        "valid_citations": sorted(list(valid_citations)),
        "hallucinated_citations": sorted(list(hallucinated_citations)),
        "citation_precision": round(precision, 2),
        "is_citation_valid": len(hallucinated_citations) == 0,
        "metadata_map": id_to_metadata
    }

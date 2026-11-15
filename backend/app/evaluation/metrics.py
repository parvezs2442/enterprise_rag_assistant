from typing import List, Dict, Any

def evaluate_retrieval_ranking(retrieved_docs: List[Any], ground_truth_page: int) -> Dict[str, float]:
    """Hit Rate @ k aur Mean Reciprocal Rank (MRR) calculate karta hai."""
    if ground_truth_page is None:
        return {"hit": 1.0, "rr": 1.0}

    for rank, doc in enumerate(retrieved_docs, start=1):
        doc_page = doc.metadata.get("page", -1) + 1  # 0-indexed to 1-indexed
        if doc_page == ground_truth_page:
            return {"hit": 1.0, "rr": 1.0 / rank}

    return {"hit": 0.0, "rr": 0.0}

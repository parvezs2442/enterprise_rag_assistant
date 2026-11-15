from typing import List
from langchain_core.documents import Document

def sanitize_untrusted_text(text: str) -> str:
    """Prompt injection defense: Null bytes aur XML instruction tags ko neutral karta hai."""
    sanitized = text.replace("\x00", "").strip()
    sanitized = sanitized.replace("<system>", "[system]").replace("</system>", "[/system]")
    sanitized = sanitized.replace("<instructions>", "[instructions]").replace("</instructions>", "[/instructions]")
    return sanitized

def format_context_with_fences(documents: List[Document]) -> str:
    """Retrieved passages ko `<untrusted_document id='DOC-X'>` XML fences mein wrap karta hai."""
    blocks = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page", 0) + 1
        score_info = f" rerank_score='{doc.metadata.get('rerank_score', 'N/A')}'" if "rerank_score" in doc.metadata else ""
        content = sanitize_untrusted_text(doc.page_content)
        block = (
            f"<untrusted_document id='DOC-{i}' source='{source}' page='{page}'{score_info}>\n"
            f"{content}\n"
            f"</untrusted_document>"
        )
        blocks.append(block)
    return "\n\n".join(blocks)

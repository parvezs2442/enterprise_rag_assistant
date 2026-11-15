from google import genai
from app.core.config import settings

def rewrite_query(user_query: str) -> str:
    """LLM query expansion: conversational query ko search-optimized keywords mein translate karta hai."""
    client = genai.Client()
    prompt = f"""You are a search query optimizer for an enterprise documentation search system.
Given the following user query, output an expanded, optimized search query with technical keywords.
Do NOT answer the question. Only output the rewritten search query in 1 concise line.

User Query: {user_query}
Rewritten Query:"""

    try:
        response = client.models.generate_content(
            model=settings.gemini_chat_model,
            contents=prompt
        )
        optimized = response.text.strip().replace('"', '')
        return optimized if optimized else user_query
    except Exception:
        return user_query

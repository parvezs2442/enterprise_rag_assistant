from google import genai
from app.core.config import settings

def generate_hyde_document(query: str) -> str:
    """HyDE: Query ka ek hypothetical ideal factual answer generate karta hai."""
    client = genai.Client()
    prompt = f"""Write a short, factual passage from a technical document that directly answers this question:
"{query}"
Do not explain, just write the passage (2-3 sentences)."""

    try:
        response = client.models.generate_content(
            model=settings.gemini_chat_model,
            contents=prompt
        )
        return response.text.strip()
    except Exception:
        return query

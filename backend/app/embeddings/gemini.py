from typing import List
from google import genai
from langchain_core.embeddings import Embeddings
from app.core.config import settings

class GeminiEmbeddingsAdapter(Embeddings):
    """
    Production-grade LangChain Embeddings adapter wrapping the Google GenAI SDK.
    Batch quota issues ko bypass karke reliable single-document embedding provide karta hai.
    """
    def __init__(self, model_name: str = settings.gemini_embed_model):
        self.client = genai.Client()
        self.model = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            response = self.client.models.embed_content(
                model=self.model,
                contents=text
            )
            embeddings.append(response.embeddings[0].values)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        response = self.client.models.embed_content(
            model=self.model,
            contents=text
        )
        return response.embeddings[0].values

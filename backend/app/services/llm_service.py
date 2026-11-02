from langchain_google_genai import ChatGoogleGenerativeAI
from app.core.config import settings


class LLMService:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.DEFAULT_MODEL
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            google_api_key=settings.GOOGLE_API_KEY
        )

    def generate_response(self, prompt: str) -> str:
        """Gemini LLM ko call karta hai aur generated response return karta hai."""
        response = self.llm.invoke(prompt)

        # Content string ya list format dono handle karna
        if isinstance(response.content, list):
            texts = [
                b.get("text", "") if isinstance(b, dict) else str(b)
                for b in response.content
            ]
            return "".join(texts)
        return str(response.content)


llm_service = LLMService()

from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.llm_service import llm_service
from app.core.config import settings

router = APIRouter()


@router.post("/", response_model=ChatResponse, summary="Prompt send to Gemini LLM")
def chat_with_gemini(request: ChatRequest):
    """
    Gemini LLM ko prompt bhejkar response generate karta hai.
    """
    try:
        reply = llm_service.generate_response(request.message)
        return ChatResponse(
            response=reply,
            model=settings.DEFAULT_MODEL
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"LLM generation me error aaya: {str(e)}"
        )

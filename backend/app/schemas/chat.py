from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        description="Prompt or question for the LLM",
        json_schema_extra={"example": "Namaste! RAG kya hota hai?"}
    )


class ChatResponse(BaseModel):
    response: str
    model: str

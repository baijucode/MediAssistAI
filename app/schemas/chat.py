from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64)
    message: str = Field(
        min_length=1,
        max_length=2000,
        pattern=r".*\S.*",
    )


class ChatResponse(BaseModel):
    session_id: str
    answer: str
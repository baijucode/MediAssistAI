from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag import answer_question

router = APIRouter(tags=["Chat"])


@router.post("/chat", response_model=ChatResponse)
def chat(
    body: ChatRequest,
    db: Annotated[Session, Depends(get_db)],
) -> ChatResponse:
    reply = answer_question(body.session_id, body.message, db)
    return ChatResponse(session_id=body.session_id, answer=reply)

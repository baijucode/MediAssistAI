import pytest
from pydantic import ValidationError

from app.schemas.chat import ChatRequest
from app.schemas.ingestion import ChunkStrategy


def test_chat_request_accepts_valid_message():
    request = ChatRequest(session_id="session-1", message="What are visiting hours?")
    assert request.session_id == "session-1"


def test_chat_request_rejects_whitespace_only_message():
    with pytest.raises(ValidationError):
        ChatRequest(session_id="session-1", message="   ")


def test_chunk_strategy_values():
    assert ChunkStrategy.FIXED.value == "fixed"
    assert ChunkStrategy.SENTENCE.value == "sentence"

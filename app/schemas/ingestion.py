from enum import Enum

from pydantic import BaseModel


class ChunkStrategy(str, Enum):
    FIXED = "fixed"
    SENTENCE = "sentence"


class IngestionResponse(BaseModel):
    """Response returned after successful document ingestion."""

    document_id: str
    filename: str
    strategy: ChunkStrategy
    chunk_count: int
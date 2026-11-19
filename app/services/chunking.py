import re
from typing import Protocol

from app.schemas.ingestion import ChunkStrategy


class Chunker(Protocol):
    """Contract: anything with chunk(text) -> list[str] counts as a Chunker."""

    def chunk(self, text: str) -> list[str]: ...


class FixedSizeChunker:
    """Fixed character windows with overlap."""

    def __init__(self, chunk_size: int = 800, overlap: int = 100) -> None:
        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk size.")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        step = self.chunk_size - self.overlap
        chunks = [text[i : i + self.chunk_size] for i in range(0, len(text), step)]
        return [c.strip() for c in chunks if c.strip()]


class SentenceChunker:
    """Groups whole sentences up to a max size, so no sentence is cut in half."""

    _SPLIT = re.compile(r"(?<=[.!?])\s+")

    def __init__(self, max_chars: int = 800) -> None:
        self.max_chars = max_chars

    def chunk(self, text: str) -> list[str]:
        chunks: list[str] = []
        current = ""

        for sentence in self._SPLIT.split(text):
            if current and len(current) + len(sentence) + 1 > self.max_chars:
                chunks.append(current.strip())
                current = ""
            current = f"{current} {sentence}".strip()

        if current:
            chunks.append(current.strip())
        return chunks


def get_chunker(strategy: ChunkStrategy) -> Chunker:
    """Factory: returns the right chunker for the chosen strategy."""
    if strategy is ChunkStrategy.FIXED:
        return FixedSizeChunker()
    return SentenceChunker()
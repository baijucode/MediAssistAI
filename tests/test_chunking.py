from app.schemas.ingestion import ChunkStrategy
from app.services.chunking import FixedSizeChunker, SentenceChunker, get_chunker


def test_fixed_chunker_returns_text_chunks():
    chunks = FixedSizeChunker(chunk_size=10, overlap=2).chunk("abcdefghijklmnop")
    assert chunks
    assert all(chunk.strip() for chunk in chunks)


def test_sentence_chunker_keeps_sentence_boundaries():
    chunks = SentenceChunker(max_chars=20).chunk("First sentence. Second sentence. Third sentence.")
    assert chunks
    assert all(chunk.endswith((".", "!", "?")) for chunk in chunks)


def test_chunker_factory_supports_both_strategies():
    assert isinstance(get_chunker(ChunkStrategy.FIXED), FixedSizeChunker)
    assert isinstance(get_chunker(ChunkStrategy.SENTENCE), SentenceChunker)

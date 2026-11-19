import uuid
from dataclasses import dataclass
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.core.config import get_settings


@dataclass(frozen=True)
class SearchHit:
    text: str
    filename: str
    score: float


class VectorStore:
    def __init__(self) -> None:
        s = get_settings()
        self._client = QdrantClient(url=s.qdrant_url, api_key=s.qdrant_api_key)
        self._collection = s.qdrant_collection
        self._dims = s.embedding_dimensions
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        if not self._client.collection_exists(self._collection):
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._dims, distance=Distance.COSINE),
            )

    def add_chunks(
        self,
        document_id: str,
        filename: str,
        chunks: list[str],
        vectors: list[list[float]],
    ) -> None:
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={
                    "document_id": document_id,
                    "filename": filename,
                    "chunk_index": index,
                    "text": chunk,
                },
            )
            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]
        self._client.upsert(collection_name=self._collection, points=points)


    def delete_document(self, document_id: str) -> None:
        """Delete every vector belonging to a document."""
        self._client.delete(
            collection_name=self._collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(value=document_id),
                    )
                ]
            ),
            wait=True,
        )

    def search(self, vector: list[float], top_k: int) -> list[SearchHit]:
        result = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=top_k,
            with_payload=True,
        )
        return [
            SearchHit(
                text=str(p.payload["text"]),
                filename=str(p.payload["filename"]),
                score=p.score,
            )
            for p in result.points
            if p.payload
        ]


@lru_cache
def get_vector_store() -> VectorStore:
    return VectorStore()
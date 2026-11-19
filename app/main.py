from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.ingestion import router as ingestion_router
from app.db import models  # noqa: F401
from app.db.session import Base, engine


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title="MediAssist AI",
    description="Backend for document ingestion and conversational RAG",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(ingestion_router)
app.include_router(chat_router)


@app.get("/")
def home() -> dict[str, str]:
    return {
        "message": "Welcome to MediAssist AI!",
        "status": "Backend is running successfully",
    }


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "Healthy",
        "message": "The backend is running smoothly.",
    }
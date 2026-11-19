import uuid
from pathlib import PurePosixPath, PureWindowsPath
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import Document
from app.db.session import get_db
from app.schemas.ingestion import ChunkStrategy, IngestionResponse
from app.services.chunking import get_chunker
from app.services.embeddings import get_embedding_service
from app.services.extraction import UnsupportedFileTypeError, extract_text
from app.services.vector_store import get_vector_store

router = APIRouter(tags=["Ingestion"])


@router.post(
    "/ingest",
    response_model=IngestionResponse,
    status_code=status.HTTP_201_CREATED,
)
def ingest_document(
    file: Annotated[
        UploadFile,
        File(description="Upload a PDF or TXT document"),
    ],
    db: Annotated[Session, Depends(get_db)],
    strategy: Annotated[ChunkStrategy, Form()] = ChunkStrategy.FIXED,
) -> IngestionResponse:

    settings = get_settings()

    max_bytes = settings.max_upload_mb * 1024 * 1024
    content = file.file.read(max_bytes + 1)

    if len(content) > max_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds the {settings.max_upload_mb} MB limit.",
        )

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    original_filename = file.filename or "upload"

    filename = (
        PureWindowsPath(original_filename).name
        if "\\" in original_filename
        else PurePosixPath(original_filename).name
    )

    if not filename or filename in {".", ".."}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename.",
        )

    try:
        text = extract_text(filename, content)
    except UnsupportedFileTypeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Unable to extract text from the uploaded document.",
        ) from exc

    if not text or not text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No readable text found. Scanned PDFs may require OCR.",
        )

    chunks = get_chunker(strategy).chunk(text)

    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Document could not be divided into text chunks.",
        )

    try:
        vectors = get_embedding_service().embed_documents(chunks)

        if len(vectors) != len(chunks):
            raise ValueError(
                "Embedding count does not match the number of chunks."
            )

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to generate document embeddings.",
        ) from exc

    document_id = str(uuid.uuid4())

    vector_store = get_vector_store()

    try:
        vector_store.add_chunks(
            document_id,
            filename,
            chunks,
            vectors,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Failed to store document vectors.",
        ) from exc

    try:
        document = Document(
            id=document_id,
            filename=filename,
            file_type=PurePosixPath(filename).suffix.lower().lstrip("."),
            strategy=strategy.value,
            chunk_count=len(chunks),
        )

        db.add(document)
        db.commit()

    except SQLAlchemyError as exc:
        db.rollback()

        try:
            vector_store.delete_document(document_id)
        except Exception:
            pass

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save document metadata.",
        ) from exc

    return IngestionResponse(
        document_id=document_id,
        filename=filename,
        strategy=strategy,
        chunk_count=len(chunks),
    )

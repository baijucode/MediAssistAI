# MediAssist AI

A FastAPI backend for document-grounded healthcare administration Q&A and interview booking.

## Features

- Upload PDF and TXT files.
- Extract text and split it using fixed-size or sentence-based chunking.
- Generate document/query embeddings with Gemini Embedding.
- Store vectors and chunk payloads in Qdrant.
- Store document metadata and confirmed interview bookings in SQLite.
- Maintain conversation history in Redis.
- Answer questions using a custom retrieval-augmented generation (RAG) pipeline.
- Collect interview details through chat and save only after explicit confirmation.

## Architecture

- **FastAPI**: REST API
- **Gemini API**: embeddings and language model
- **Qdrant**: vector storage and similarity search
- **Redis**: session-based chat history
- **SQLite + SQLAlchemy**: document metadata and interview bookings

## Setup (Windows PowerShell)

1. Create and activate a virtual environment:

   ```powershell
   py -3.12 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Create your local environment file:

   ```powershell
   Copy-Item .env.example .env
   ```

   Open `.env` and add your own `GEMINI_API_KEY`. Never commit `.env`.

4. Start Qdrant and Redis:

   ```powershell
   docker compose up -d
   ```

5. Start the API:

   ```powershell
   uvicorn app.main:app --reload
   ```

6. Open the interactive API documentation:

   `http://127.0.0.1:8000/docs`

## API endpoints

- `GET /` — welcome/status
- `GET /health` — health check
- `POST /ingest` — upload a PDF/TXT file; multipart fields: `file`, optional `strategy` (`fixed` or `sentence`)
- `POST /chat` — chat; JSON fields: `session_id`, `message`

## Interview booking flow

In `/chat`, ask to book an interview. The assistant collects the applicant's name, email, preferred date, and time, summarises the details, and asks for explicit confirmation. After confirmation, the booking is saved in SQLite.

The application records the requested slot; it does **not** connect to a calendar or independently verify interviewer availability. Do not describe a saved request as a verified available slot.

## Run tests

```powershell
pytest -q
```

## Important notes

- Scanned PDFs require OCR; this version extracts embedded PDF text only.
- Healthcare answers are restricted to uploaded document context and are not medical diagnosis or treatment advice.
- Configure `embedding_dimensions` consistently with the embedding output and Qdrant collection. If changing dimensions after creating the collection, recreate the Qdrant collection.
- Local `.env`, `.venv`, SQLite databases, and data files are excluded from Git.

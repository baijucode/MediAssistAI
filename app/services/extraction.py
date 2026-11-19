from io import BytesIO
from pathlib import PurePath

from pypdf import PdfReader

ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".txt"})


class UnsupportedFileTypeError(ValueError):
    """Raised when the uploaded file is not a .pdf or .txt."""


def extract_text(filename: str, content: bytes) -> str:
    suffix = PurePath(filename).suffix.lower()

    if suffix == ".txt":
        return content.decode("utf-8", errors="ignore").strip()

    if suffix == ".pdf":
        reader = PdfReader(BytesIO(content))
        pages = (page.extract_text() or "" for page in reader.pages)
        return "\n".join(pages).strip()

    raise UnsupportedFileTypeError(f"Unsupported file type: {suffix}")
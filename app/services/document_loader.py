from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import docx
import fitz

from app.config import Settings
from app.schemas import DocumentChunk

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[str]:
    cleaned_text = _clean_text(text)
    if not cleaned_text:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be greater than or equal to 0")

    paragraphs = _split_paragraphs(cleaned_text)
    if not paragraphs:
        return []

    chunks: list[str] = []
    current_paragraphs: list[str] = []

    for paragraph in paragraphs:
        if len(paragraph) > chunk_size:
            if current_paragraphs:
                chunks.append("\n\n".join(current_paragraphs).strip())
                current_paragraphs = []

            chunks.extend(_split_long_text(paragraph, chunk_size, chunk_overlap))
            continue

        candidate_parts = current_paragraphs + [paragraph]
        candidate_text = "\n\n".join(candidate_parts).strip()
        if not current_paragraphs or len(candidate_text) <= chunk_size:
            current_paragraphs = candidate_parts
            continue

        chunks.append("\n\n".join(current_paragraphs).strip())
        overlap_parts = _build_overlap_paragraphs(current_paragraphs, chunk_overlap)
        retry_parts = overlap_parts + [paragraph] if overlap_parts else [paragraph]
        retry_text = "\n\n".join(retry_parts).strip()

        if len(retry_text) <= chunk_size:
            current_paragraphs = retry_parts
        else:
            current_paragraphs = [paragraph]

    if current_paragraphs:
        chunks.append("\n\n".join(current_paragraphs).strip())

    return [chunk for chunk in chunks if chunk.strip()]


def load_documents_from_dir(directory: str) -> list[dict]:
    directory_path = Path(directory)
    documents: list[dict] = []

    if not directory_path.exists() or not directory_path.is_dir():
        logger.warning("Directory does not exist or is not a directory: %s", directory)
        return documents

    for file_path in sorted(directory_path.iterdir()):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            text = _load_text(file_path)
            cleaned_text = _clean_text(text)
            if not cleaned_text:
                logger.warning("Skip empty document: %s", file_path.as_posix())
                continue

            documents.append(
                {
                    "source": file_path.name,
                    "text": cleaned_text,
                }
            )
        except Exception as exc:
            logger.warning("Failed to load document %s: %s", file_path.as_posix(), exc)

    return documents


class DocumentLoader:
    SUPPORTED_EXTENSIONS = SUPPORTED_EXTENSIONS

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def is_supported_file(self, file_path: Path) -> bool:
        return file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS

    def list_supported_files(self, directory: Path) -> list[Path]:
        if not directory.exists():
            return []
        return sorted(
            file_path
            for file_path in directory.iterdir()
            if file_path.is_file() and self.is_supported_file(file_path)
        )

    def load_text(self, file_path: Path) -> str:
        return _load_text(file_path)

    def load_documents_from_dir(self, directory: str) -> list[dict]:
        return load_documents_from_dir(directory)

    def load_and_split(self, file_path: Path) -> list[DocumentChunk]:
        text = _clean_text(self.load_text(file_path))
        if not text:
            return []

        chunks: list[DocumentChunk] = []
        for chunk_index, chunk_text in enumerate(
            split_text(
                text=text,
                chunk_size=self.settings.chunk_size,
                chunk_overlap=self.settings.chunk_overlap,
            )
        ):
            chunk_id = self._build_chunk_id(file_path.name, chunk_index, chunk_text)
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    source=file_path.name,
                    text=chunk_text,
                    metadata={
                        "source": file_path.name,
                        "path": file_path.as_posix(),
                        "chunk_index": chunk_index,
                    },
                )
            )
        return chunks

    def _build_chunk_id(self, file_name: str, chunk_index: int, chunk_text: str) -> str:
        digest = hashlib.md5(chunk_text.encode("utf-8")).hexdigest()[:10]
        return f"{file_name}-{chunk_index}-{digest}"


def _load_text(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(file_path)
    if suffix == ".docx":
        return _load_docx(file_path)
    if suffix == ".txt":
        return _load_txt(file_path)
    raise ValueError(f"Unsupported file type: {suffix}")


def _load_pdf(file_path: Path) -> str:
    pages: list[str] = []
    with fitz.open(file_path) as document:
        for page in document:
            page_text = page.get_text("text")
            if page_text:
                pages.append(page_text)
    return "\n".join(pages)


def _load_docx(file_path: Path) -> str:
    document = docx.Document(file_path)
    paragraphs = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
    return "\n".join(paragraphs)


def _load_txt(file_path: Path) -> str:
    try:
        return file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return file_path.read_text(encoding="gbk", errors="ignore")


def _clean_text(text: str) -> str:
    normalized = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.strip() for line in normalized.splitlines()]

    paragraphs: list[str] = []
    buffer: list[str] = []

    for line in lines:
        if line:
            buffer.append(line)
            continue

        if buffer:
            paragraphs.append(" ".join(buffer).strip())
            buffer = []

    if buffer:
        paragraphs.append(" ".join(buffer).strip())

    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph).strip()


def _split_paragraphs(text: str) -> list[str]:
    return [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]


def _build_overlap_paragraphs(paragraphs: list[str], chunk_overlap: int) -> list[str]:
    if chunk_overlap <= 0:
        return []

    selected: list[str] = []
    current_length = 0

    for paragraph in reversed(paragraphs):
        additional_length = len(paragraph)
        if selected:
            additional_length += 2

        if selected and current_length + additional_length > chunk_overlap:
            break

        if not selected and len(paragraph) > chunk_overlap:
            return []

        selected.insert(0, paragraph)
        current_length += additional_length

    return selected


def _split_long_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    text_length = len(text)
    min_break_point = max(1, int(chunk_size * 0.6))

    while start < text_length:
        end = min(text_length, start + chunk_size)

        if end < text_length:
            break_positions = [
                text.rfind("\n\n", start, end),
                text.rfind("\u3002", start, end),
                text.rfind("\uff01", start, end),
                text.rfind("\uff1f", start, end),
                text.rfind(".", start, end),
                text.rfind("!", start, end),
                text.rfind("?", start, end),
                text.rfind(";", start, end),
                text.rfind("\uff1b", start, end),
                text.rfind(" ", start, end),
            ]
            best_break = max(break_positions)
            if best_break >= start + min_break_point:
                end = best_break + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        next_start = max(end - chunk_overlap, start + 1)
        while next_start < text_length and text[next_start].isspace():
            next_start += 1
        start = next_start

    return chunks

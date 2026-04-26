from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    app_name: str
    llm_configured: bool


class UploadResponse(BaseModel):
    message: str
    filename: str
    saved_path: str
    size: int


class DocumentFileInfo(BaseModel):
    filename: str
    path: str
    indexed: bool = False


class DocumentListResponse(BaseModel):
    files: list[DocumentFileInfo]


class IndexRequest(BaseModel):
    filenames: list[str] = Field(default_factory=list)


class IndexResponse(BaseModel):
    message: str
    indexed_files: list[str]
    skipped_files: list[str]
    total_chunks: int


class SourceChunk(BaseModel):
    source: str
    chunk_id: str
    score: float
    content: str


class QARequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=4, ge=1, le=10)


class QAResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=4, ge=1, le=10)


class ChatReference(BaseModel):
    source: str
    chunk_index: int
    score: float


class ChatResponse(BaseModel):
    answer: str
    references: list[ChatReference]
    contexts: list[str]


class DiagnoseRequest(BaseModel):
    symptom: str = Field(..., min_length=1)
    device_type: str | None = None
    top_k: int = Field(default=4, ge=1, le=10)


class DiagnoseReference(BaseModel):
    source: str
    chunk_index: int
    score: float


class DiagnoseResponse(BaseModel):
    possible_causes: list[str]
    troubleshooting_steps: list[str]
    safety_notes: list[str]
    references: list[DiagnoseReference]


class DocumentChunk(BaseModel):
    chunk_id: str
    source: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)

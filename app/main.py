from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.schemas import (
    ChatRequest,
    ChatResponse,
    DiagnoseRequest,
    DiagnoseResponse,
    DocumentFileInfo,
    DocumentListResponse,
    HealthResponse,
    IndexRequest,
    IndexResponse,
    QARequest,
    QAResponse,
    SourceChunk,
    UploadResponse,
)
from app.services.diagnosis import DiagnosisService
from app.services.document_loader import DocumentLoader
from app.services.llm_client import LLMClient
from app.services.rag_chain import RAGChain
from app.services.vector_store import VectorStore

settings = get_settings()
document_loader = DocumentLoader(settings)
vector_store = VectorStore(settings)
llm_client = LLMClient(settings)
rag_chain = RAGChain(vector_store=vector_store, llm_client=llm_client)
diagnosis_service = DiagnosisService(vector_store=vector_store, llm_client=llm_client)

app = FastAPI(title=settings.app_name, version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        llm_configured=llm_client.enabled,
    )


@app.get("/documents", response_model=DocumentListResponse)
def list_documents() -> DocumentListResponse:
    indexed_sources = vector_store.list_indexed_sources()
    files = [
        DocumentFileInfo(
            filename=file_path.name,
            path=file_path.as_posix(),
            indexed=file_path.name in indexed_sources,
        )
        for file_path in document_loader.list_supported_files(settings.upload_path)
    ]
    return DocumentListResponse(files=files)


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="文件名不能为空。")

    file_path = settings.upload_path / filename
    if not document_loader.is_supported_file(file_path):
        raise HTTPException(status_code=400, detail="仅支持 pdf、docx、txt 文件。")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="上传文件内容为空。")

    file_path.write_bytes(content)
    return UploadResponse(
        message="文件上传成功。",
        filename=filename,
        saved_path=file_path.as_posix(),
        size=len(content),
    )


@app.post("/documents/index", response_model=IndexResponse)
def index_documents(request: IndexRequest) -> IndexResponse:
    requested_files = request.filenames or [
        file_path.name
        for file_path in document_loader.list_supported_files(settings.upload_path)
    ]
    if not requested_files:
        raise HTTPException(status_code=400, detail="未找到可索引的文档。")

    indexed_files: list[str] = []
    skipped_files: list[str] = []
    total_chunks = 0

    for filename in requested_files:
        file_path = settings.upload_path / Path(filename).name
        if not file_path.exists() or not document_loader.is_supported_file(file_path):
            skipped_files.append(filename)
            continue

        try:
            document_chunks = document_loader.load_and_split(file_path)
            chunk_payload = [
                {
                    "id": chunk.chunk_id,
                    "text": chunk.text,
                    "source": chunk.source,
                    "chunk_index": int(chunk.metadata.get("chunk_index", index)),
                }
                for index, chunk in enumerate(document_chunks)
            ]
            inserted_count = vector_store.upsert_documents(chunk_payload)
        except Exception as exc:
            skipped_files.append(f"{filename} ({exc})")
            continue

        if inserted_count == 0:
            skipped_files.append(f"{filename} (empty)")
            continue

        total_chunks += inserted_count
        indexed_files.append(file_path.name)

    if not indexed_files:
        raise HTTPException(status_code=400, detail="没有成功建立索引的文档。")

    return IndexResponse(
        message="索引建立完成。",
        indexed_files=indexed_files,
        skipped_files=skipped_files,
        total_chunks=total_chunks,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    result = rag_chain.answer_question(query=request.query, top_k=request.top_k)
    return ChatResponse(**result)


@app.post("/qa", response_model=QAResponse)
def ask_question(request: QARequest) -> QAResponse:
    result = rag_chain.answer_question(query=request.question, top_k=request.top_k)
    sources = [
        SourceChunk(
            source=reference["source"],
            chunk_id=f"{reference['source']}-{reference['chunk_index']}",
            score=reference["score"],
            content=context,
        )
        for reference, context in zip(result["references"], result["contexts"])
    ]
    return QAResponse(answer=result["answer"], sources=sources)


@app.post("/diagnose", response_model=DiagnoseResponse)
def diagnose(request: DiagnoseRequest) -> DiagnoseResponse:
    result = diagnosis_service.diagnose_issue(
        symptom=request.symptom,
        device_type=request.device_type,
        top_k=request.top_k,
    )
    return DiagnoseResponse(**result)

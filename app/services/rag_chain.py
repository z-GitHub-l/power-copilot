from __future__ import annotations

from typing import Any

from app.services.llm_client import LLMClient
from app.services.vector_store import VectorStore


class RAGChain:
    def __init__(self, vector_store: VectorStore, llm_client: LLMClient) -> None:
        self.vector_store = vector_store
        self.llm_client = llm_client

    def answer_question(self, query: str, top_k: int = 4) -> dict[str, Any]:
        retrieved_chunks = self.vector_store.similarity_search(query=query, top_k=top_k)
        valid_chunks = [chunk for chunk in retrieved_chunks if chunk.get("text", "").strip()]

        if not valid_chunks:
            return {
                "answer": "当前知识库中没有足够信息来回答这个问题，请先上传并建立相关文档索引。",
                "references": [],
                "contexts": [],
            }

        contexts = [chunk["text"] for chunk in valid_chunks]
        references = [
            {
                "source": chunk["source"],
                "chunk_index": int(chunk["chunk_index"]),
                "score": float(chunk["score"]),
            }
            for chunk in valid_chunks
        ]

        context_text = self._build_context(valid_chunks)
        system_prompt = (
            "你是电力运维智能助手。"
            "请基于提供的知识库片段回答问题，保持专业、简洁、可执行。"
            "如果上下文不足，请明确说明知识库信息不足，不要编造。"
        )
        user_prompt = (
            f"问题：{query}\n\n"
            f"知识库上下文：\n{context_text}\n\n"
            "请给出简明结论，并在必要时给出建议的排查或执行步骤。"
        )

        if not self.llm_client.enabled:
            answer = self._build_local_answer(query=query, chunks=valid_chunks)
        else:
            try:
                answer = self.llm_client.chat_completion(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=0.2,
                )
            except Exception as exc:
                answer = self._build_local_answer(
                    query=query,
                    chunks=valid_chunks,
                    note=f"LLM 调用失败，已回退为本地摘要。错误信息：{exc}",
                )

        return {
            "answer": answer,
            "references": references,
            "contexts": contexts,
        }

    def _build_context(self, chunks: list[dict[str, Any]]) -> str:
        parts = []
        for index, chunk in enumerate(chunks, start=1):
            parts.append(
                f"[参考{index}] source={chunk['source']} "
                f"chunk_index={chunk['chunk_index']} "
                f"score={chunk['score']}\n{chunk['text']}"
            )
        return "\n\n".join(parts)

    def _build_local_answer(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        note: str | None = None,
    ) -> str:
        lines = [
            "当前未使用外部 LLM，以下为基于检索结果的本地摘要。",
            f"问题：{query}",
            "",
            "参考要点：",
        ]
        for chunk in chunks:
            lines.append(f"- [{chunk['source']}] {chunk['text'][:220]}")

        if note:
            lines.extend(["", note])

        return "\n".join(lines).strip()

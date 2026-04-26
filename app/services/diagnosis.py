from __future__ import annotations

import json
from typing import Any

from app.services.llm_client import LLMClient
from app.services.vector_store import VectorStore


class DiagnosisService:
    def __init__(self, vector_store: VectorStore, llm_client: LLMClient) -> None:
        self.vector_store = vector_store
        self.llm_client = llm_client

    def diagnose_issue(
        self,
        symptom: str,
        device_type: str | None = None,
        top_k: int = 4,
    ) -> dict[str, Any]:
        query = self._build_query(symptom=symptom, device_type=device_type)
        retrieved_chunks = self._retrieve_contexts(query=query, top_k=top_k)
        references = self._build_references(retrieved_chunks)

        if not retrieved_chunks:
            return self._build_insufficient_result(references=references)

        context_text = self._build_context_text(retrieved_chunks)
        system_prompt, user_prompt = self._build_prompts(
            symptom=symptom,
            device_type=device_type,
            context_text=context_text,
        )

        if not self.llm_client.enabled:
            return self._build_fallback_result(
                symptom=symptom,
                device_type=device_type,
                references=references,
                reason="当前未配置 LLM 服务，以下结果基于检索证据和本地规则生成。",
                has_context=True,
            )

        try:
            llm_text = self.llm_client.chat_completion(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=0.2,
            )
            parsed = self._parse_llm_output(llm_text)
            parsed["references"] = references
            return parsed
        except Exception as exc:
            return self._build_fallback_result(
                symptom=symptom,
                device_type=device_type,
                references=references,
                reason=f"LLM 调用或解析失败，已回退到本地模板。错误信息：{exc}",
                has_context=True,
            )

    def _build_query(self, symptom: str, device_type: str | None) -> str:
        parts = [symptom.strip()]
        if device_type and device_type.strip():
            parts.append(device_type.strip())
        return " ".join(part for part in parts if part)

    def _retrieve_contexts(self, query: str, top_k: int) -> list[dict[str, Any]]:
        return self.vector_store.similarity_search(query=query, top_k=top_k)

    def _build_references(self, retrieved_chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "source": str(chunk.get("source", "unknown")),
                "chunk_index": int(chunk.get("chunk_index", 0)),
                "score": float(chunk.get("score", 0.0)),
            }
            for chunk in retrieved_chunks
        ]

    def _build_context_text(self, retrieved_chunks: list[dict[str, Any]]) -> str:
        parts: list[str] = []
        for index, chunk in enumerate(retrieved_chunks, start=1):
            parts.append(
                f"[证据{index}] "
                f"source={chunk.get('source', 'unknown')} "
                f"chunk_index={chunk.get('chunk_index', 0)} "
                f"score={chunk.get('score', 0.0)}\n"
                f"{chunk.get('text', '')}"
            )
        return "\n\n".join(parts)

    def _build_prompts(
        self,
        symptom: str,
        device_type: str | None,
        context_text: str,
    ) -> tuple[str, str]:
        system_prompt = (
            "你是电力运维故障诊断助手。"
            "请严格基于提供的知识库证据给出专业、简洁、可执行的诊断建议。"
            "你的输出必须是 JSON，且只能包含以下四个字段："
            "possible_causes、troubleshooting_steps、safety_notes、references。"
            "前三个字段必须是字符串数组，references 返回空数组即可。"
            "如果证据不足，请在数组中明确写出“当前知识库信息不足，需要补充现场信息/图纸/告警记录”等提示，不要编造事实。"
        )
        user_prompt = (
            f"故障现象：{symptom.strip()}\n"
            f"设备类型：{(device_type or '未提供').strip() if device_type else '未提供'}\n\n"
            f"知识库证据：\n{context_text}\n\n"
            "请输出 JSON，例如："
            '{"possible_causes":["..."],'
            '"troubleshooting_steps":["..."],'
            '"safety_notes":["..."],'
            '"references":[]}'
        )
        return system_prompt, user_prompt

    def _parse_llm_output(self, llm_text: str) -> dict[str, Any]:
        payload_text = llm_text.strip()
        if payload_text.startswith("```"):
            lines = payload_text.splitlines()
            if len(lines) >= 3:
                payload_text = "\n".join(lines[1:-1]).strip()

        data = json.loads(payload_text)
        if not isinstance(data, dict):
            raise ValueError("LLM output is not a JSON object.")

        normalized = {
            "possible_causes": self._normalize_string_list(data.get("possible_causes")),
            "troubleshooting_steps": self._normalize_string_list(
                data.get("troubleshooting_steps")
            ),
            "safety_notes": self._normalize_string_list(data.get("safety_notes")),
            "references": [],
        }

        if not normalized["possible_causes"]:
            normalized["possible_causes"] = ["当前知识库信息不足，暂无法准确判断可能原因。"]
        if not normalized["troubleshooting_steps"]:
            normalized["troubleshooting_steps"] = ["请补充设备类型、告警记录和现场现象后再进一步分析。"]
        if not normalized["safety_notes"]:
            normalized["safety_notes"] = ["排查前请确认已按规程执行停送电、验电和监护措施。"]

        return normalized

    def _normalize_string_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    def _build_insufficient_result(self, references: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "possible_causes": [
                "当前知识库中没有检索到足够相关的故障案例或设备资料，暂无法准确判断可能原因。"
            ],
            "troubleshooting_steps": [
                "请补充更具体的故障现象、设备类型、告警报码、动作记录或运行工况后再进行诊断。"
            ],
            "safety_notes": [
                "在信息不足的情况下不要贸然操作设备，需先落实停送电审批、验电、接地和现场监护措施。"
            ],
            "references": references,
        }

    def _build_fallback_result(
        self,
        symptom: str,
        device_type: str | None,
        references: list[dict[str, Any]],
        reason: str,
        has_context: bool,
    ) -> dict[str, Any]:
        symptom_text = f"{symptom} {device_type or ''}".lower()

        possible_causes = []
        troubleshooting_steps = []
        safety_notes = [
            "排查前确认工作票、操作票及停送电流程完整有效。",
            "涉及高压设备时，必须执行验电、接地、挂牌和专人监护措施。",
        ]

        if not has_context:
            possible_causes.append("当前知识库信息不足，无法形成可靠诊断结论。")
            troubleshooting_steps.append("请先补充设备说明书、运行记录、告警信息或检修文档。")
        else:
            possible_causes.append("已检索到部分相关资料，但自动结构化分析未完全成功，以下为保守建议。")
            troubleshooting_steps.append("优先核对检索到的参考资料与当前故障现象是否一致。")

        if any(keyword in symptom_text for keyword in ["跳闸", "trip", "断路器", "breaker"]):
            possible_causes.extend(
                [
                    "保护动作、瞬时过流或控制回路异常导致设备跳闸。",
                    "断路器本体机构异常或二次回路接触不良。",
                ]
            )
            troubleshooting_steps.extend(
                [
                    "检查保护装置事件记录、SOE 记录和动作先后顺序。",
                    "检查断路器储能状态、控制电源和分合闸回路。",
                ]
            )

        if any(keyword in symptom_text for keyword in ["温度", "过热", "发热", "hot"]):
            possible_causes.extend(
                [
                    "接点接触电阻增大导致局部温升异常。",
                    "长期过载、散热不良或风冷系统异常。",
                ]
            )
            troubleshooting_steps.extend(
                [
                    "核对负荷电流、环境温度和通风散热条件。",
                    "排查母排连接处、接线端子和风机运行状态。",
                ]
            )

        if any(keyword in symptom_text for keyword in ["告警", "报警", "alarm", "异常"]):
            possible_causes.extend(
                [
                    "传感器异常、通信中断或阈值越限导致告警上送。",
                    "设备内部参数偏离正常范围触发监控告警。",
                ]
            )
            troubleshooting_steps.extend(
                [
                    "核对告警点位、测点值、通信链路和主站记录。",
                    "对比现场表计、保护装置和监控系统读数是否一致。",
                ]
            )

        possible_causes.append(reason)
        troubleshooting_steps.append("若现场风险不明确，请先停止进一步操作并上报值班负责人。")

        return {
            "possible_causes": self._deduplicate(possible_causes),
            "troubleshooting_steps": self._deduplicate(troubleshooting_steps),
            "safety_notes": self._deduplicate(safety_notes),
            "references": references,
        }

    def _deduplicate(self, items: list[str]) -> list[str]:
        seen: set[str] = set()
        results: list[str] = []
        for item in items:
            value = item.strip()
            if not value or value in seen:
                continue
            seen.add(value)
            results.append(value)
        return results

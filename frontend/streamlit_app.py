from __future__ import annotations

import os
from typing import Any

import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

DEFAULT_BACKEND_URL = (
    os.getenv("BACKEND_API_URL")
    or os.getenv("FRONTEND_API_BASE_URL")
    or "http://127.0.0.1:8000"
)
REQUEST_TIMEOUT = 120


def request_json(
    method: str,
    base_url: str,
    path: str,
    **kwargs: Any,
) -> dict[str, Any]:
    url = f"{base_url.rstrip('/')}{path}"
    try:
        response = requests.request(
            method=method,
            url=url,
            timeout=REQUEST_TIMEOUT,
            **kwargs,
        )
    except requests.Timeout as exc:
        raise RuntimeError(f"请求超时：{url}") from exc
    except requests.RequestException as exc:
        raise RuntimeError(f"请求失败：{exc}") from exc

    if not response.ok:
        detail = _extract_error_message(response)
        raise RuntimeError(f"接口调用失败（{response.status_code}）：{detail}")

    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError("接口返回的不是有效 JSON。") from exc


def _extract_error_message(response: requests.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or "未知错误"

    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, str):
            return detail
        if isinstance(detail, list):
            return "; ".join(str(item) for item in detail)
    return str(payload)


def load_documents(base_url: str) -> list[dict[str, Any]]:
    result = request_json("GET", base_url, "/documents")
    return result.get("files", [])


def render_references(references: list[dict[str, Any]]) -> None:
    if not references:
        st.info("暂无参考来源。")
        return

    for reference in references:
        source = reference.get("source", "unknown")
        chunk_index = reference.get("chunk_index", 0)
        score = reference.get("score", 0.0)
        st.write(f"- 来源：{source} | 分块：{chunk_index} | 分数：{score}")


def main() -> None:
    st.set_page_config(page_title="power-copilot", layout="wide")
    st.title("power-copilot")
    st.caption("电力运维智能助手 MVP：文档上传、建立索引、知识问答、故障诊断")

    if "documents" not in st.session_state:
        st.session_state.documents = []
    if "chat_result" not in st.session_state:
        st.session_state.chat_result = None
    if "diagnose_result" not in st.session_state:
        st.session_state.diagnose_result = None

    backend_url = st.sidebar.text_input("FastAPI 地址", value=DEFAULT_BACKEND_URL)

    col_a, col_b = st.sidebar.columns(2)
    if col_a.button("检查服务", use_container_width=True):
        with st.sidebar:
            with st.spinner("检查后端状态中..."):
                try:
                    health = request_json("GET", backend_url, "/health")
                    st.success(
                        f"服务正常：{health['app_name']} | LLM 已配置：{health['llm_configured']}"
                    )
                except Exception as exc:
                    st.error(str(exc))

    if col_b.button("刷新文档", use_container_width=True):
        with st.sidebar:
            with st.spinner("刷新文档列表中..."):
                try:
                    st.session_state.documents = load_documents(backend_url)
                    st.success("文档列表已刷新。")
                except Exception as exc:
                    st.error(str(exc))

    if not st.session_state.documents:
        try:
            st.session_state.documents = load_documents(backend_url)
        except Exception:
            st.session_state.documents = []

    st.subheader("1. 文档上传")
    uploaded_files = st.file_uploader(
        "选择一个或多个 pdf、docx、txt 文件",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )
    if st.button("上传文档", use_container_width=True):
        if not uploaded_files:
            st.warning("请先选择至少一个文件。")
        else:
            upload_results: list[str] = []
            with st.spinner("上传文件中..."):
                for uploaded_file in uploaded_files:
                    files = {
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            uploaded_file.type or "application/octet-stream",
                        )
                    }
                    try:
                        result = request_json("POST", backend_url, "/documents/upload", files=files)
                        upload_results.append(
                            f"{result['filename']}：上传成功，大小 {result['size']} 字节"
                        )
                    except Exception as exc:
                        upload_results.append(f"{uploaded_file.name}：上传失败，{exc}")

            for item in upload_results:
                if "上传成功" in item:
                    st.success(item)
                else:
                    st.error(item)

            try:
                st.session_state.documents = load_documents(backend_url)
            except Exception as exc:
                st.warning(f"上传完成，但刷新文档列表失败：{exc}")

    if st.session_state.documents:
        st.dataframe(st.session_state.documents, use_container_width=True)
    else:
        st.info("当前还没有可用文档。")

    st.subheader("2. 建立索引")
    if st.button("为当前已上传文档建立索引", use_container_width=True):
        with st.spinner("建立索引中，请稍候..."):
            try:
                result = request_json(
                    "POST",
                    backend_url,
                    "/documents/index",
                    json={"filenames": []},
                )
                st.success(
                    f"{result['message']} 已处理 {len(result['indexed_files'])} 个文件，共 {result['total_chunks']} 个分块。"
                )
                if result.get("indexed_files"):
                    st.write("已建立索引的文件：")
                    for file_name in result["indexed_files"]:
                        st.write(f"- {file_name}")
                if result.get("skipped_files"):
                    st.warning("以下文件被跳过：")
                    for file_name in result["skipped_files"]:
                        st.write(f"- {file_name}")
                st.session_state.documents = load_documents(backend_url)
            except Exception as exc:
                st.error(str(exc))

    st.subheader("3. 知识问答")
    question = st.text_area(
        "请输入问题",
        height=140,
        placeholder="例如：变压器温升异常时，现场应优先检查哪些内容？",
    )
    qa_top_k = st.slider("问答检索条数", min_value=1, max_value=8, value=4)
    if st.button("开始问答", use_container_width=True):
        if not question.strip():
            st.warning("请输入问题后再提交。")
        else:
            with st.spinner("生成回答中..."):
                try:
                    st.session_state.chat_result = request_json(
                        "POST",
                        backend_url,
                        "/chat",
                        json={"query": question, "top_k": qa_top_k},
                    )
                except Exception as exc:
                    st.session_state.chat_result = None
                    st.error(str(exc))

    if st.session_state.chat_result:
        chat_result = st.session_state.chat_result
        st.markdown("**回答**")
        st.write(chat_result.get("answer", ""))

        st.markdown("**参考来源**")
        render_references(chat_result.get("references", []))

        contexts = chat_result.get("contexts", [])
        if contexts:
            with st.expander("查看检索到的上下文"):
                for index, context in enumerate(contexts, start=1):
                    st.markdown(f"**上下文 {index}**")
                    st.write(context)

    st.subheader("4. 故障诊断")
    diag_col1, diag_col2 = st.columns(2)
    with diag_col1:
        symptom = st.text_area(
            "故障现象",
            height=120,
            placeholder="例如：断路器跳闸并伴随保护告警",
        )
    with diag_col2:
        device_type = st.text_input(
            "设备类型",
            placeholder="例如：断路器、变压器、开关柜",
        )

    diag_top_k = st.slider("诊断检索条数", min_value=1, max_value=8, value=4, key="diag_top_k")
    if st.button("开始诊断", use_container_width=True):
        if not symptom.strip():
            st.warning("请输入故障现象后再提交。")
        else:
            with st.spinner("分析诊断中..."):
                try:
                    st.session_state.diagnose_result = request_json(
                        "POST",
                        backend_url,
                        "/diagnose",
                        json={
                            "symptom": symptom,
                            "device_type": device_type or None,
                            "top_k": diag_top_k,
                        },
                    )
                except Exception as exc:
                    st.session_state.diagnose_result = None
                    st.error(str(exc))

    if st.session_state.diagnose_result:
        diagnose_result = st.session_state.diagnose_result

        st.markdown("**可能原因**")
        for item in diagnose_result.get("possible_causes", []):
            st.write(f"- {item}")

        st.markdown("**建议排查步骤**")
        for item in diagnose_result.get("troubleshooting_steps", []):
            st.write(f"- {item}")

        st.markdown("**安全注意事项**")
        for item in diagnose_result.get("safety_notes", []):
            st.write(f"- {item}")

        st.markdown("**参考来源**")
        render_references(diagnose_result.get("references", []))


if __name__ == "__main__":
    main()

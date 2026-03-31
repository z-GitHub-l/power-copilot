# power-copilot

`power-copilot` 是一个可本地运行的最小可行项目，用于演示“电力运维智能助手（LLM + RAG + Agent）”的 MVP 版本。

后端使用 FastAPI，前端使用 Streamlit，向量数据库使用本地持久化的 Chroma。当前支持的核心能力包括：

- 文档上传
- 文档索引
- 知识问答
- 轻量故障诊断

## 项目结构

```text
power-copilot/
├── app/
│   ├── config.py
│   ├── main.py
│   ├── schemas.py
│   └── services/
│       ├── diagnosis.py
│       ├── document_loader.py
│       ├── llm_client.py
│       ├── rag_chain.py
│       └── vector_store.py
├── data/
│   ├── chroma/
│   └── uploads/
├── frontend/
│   └── streamlit_app.py
├── .env.example
├── README.md
└── requirements.txt
```

## 环境变量

复制 `.env.example` 为 `.env`，然后根据需要修改：

```env
APP_NAME=power-copilot
BACKEND_HOST=127.0.0.1
BACKEND_PORT=8000
FRONTEND_API_BASE_URL=http://127.0.0.1:8000

UPLOAD_DIR=data/uploads
CHROMA_DIR=data/chroma
COLLECTION_NAME=power_copilot_documents
CHUNK_SIZE=800
CHUNK_OVERLAP=120

LLM_API_KEY=
LLM_BASE_URL=
LLM_MODEL=
LLM_TIMEOUT=60
LLM_TEMPERATURE=0.2
```

说明：

- `FRONTEND_API_BASE_URL` 用于给 Streamlit 页面提供默认后端地址
- `LLM_BASE_URL` 需要填写完整的 Chat Completions 接口地址，例如 `http://127.0.0.1:8001/v1/chat/completions`
- 若未配置 LLM，问答和诊断会自动回退到本地提示或本地规则结果

## 本地启动

### 1. 激活环境

```powershell
conda activate power_copilot
```

### 2. 安装依赖

```powershell
pip install -r requirements.txt
```

### 3. 启动 FastAPI 后端

在项目根目录执行：

```powershell
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

启动后可访问：

- FastAPI 文档：`http://127.0.0.1:8000/docs`
- 健康检查：`http://127.0.0.1:8000/health`

### 4. 启动 Streamlit 前端

新开一个终端，在项目根目录执行：

```powershell
streamlit run frontend/streamlit_app.py
```

默认页面地址：

- `http://127.0.0.1:8501`

## 接口说明

### `POST /documents/upload`

上传单个文档文件，支持 `pdf`、`docx`、`txt`。

### `POST /documents/index`

为当前上传目录中的文档建立索引。

### `POST /chat`

执行 RAG 问答，返回：

- `answer`
- `references`
- `contexts`

### `POST /diagnose`

执行轻量故障诊断，返回：

- `possible_causes`
- `troubleshooting_steps`
- `safety_notes`
- `references`

## 说明

- LLM 调用层使用 `requests`，不依赖任何厂商 SDK
- Chroma 使用 `data/chroma` 做本地持久化
- 当前 embedding 是本地轻量 stub，后续可以替换为真实 embedding API 或本地模型
- 所有文件路径均使用相对路径，兼容 Windows 本地运行

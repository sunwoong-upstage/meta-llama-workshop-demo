# 고객 지원 에이전트 데모

LangGraph와 로컬 LLM 분류를 활용한 실제 고객 지원 에이전트 시스템

## 사전 요구사항

### Ollama 설치 (로컬 LLM 필수)

[Ollama](https://ollama.com/download)

### API 키

다음 API 키가 필요합니다:
- **Upstage API**: [여기서 발급](https://console.upstage.ai/)
- **Tavily API**: [여기서 발급](https://tavily.com/)

## Quick Start

### 1. 환경 설정

자세한 설정 방법은 `환경설정 가이드.md`를 참조하세요.

- uv 설치
  
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

- 클론 및 설정

```
git clone git@github.com:sunwoong-upstage/meta-llama-workshop-demo.git
cd meta-llama-workshop-demo
uv init
uv venv --python 3.11
uv add langgraph langchain-core openai==1.52.2 faiss-cpu numpy tavily-python python-dotenv "httpx<0.28.0" jupyter ipykernel langchain-upstage langchain langgraph-cli[inmem]
```

### 2. 환경 변수

`.env` 파일 생성:
```env
UPSTAGE_API_KEY=your_upstage_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
LOCAL_LLM_MODEL=llama3.2:3b
```

### 3. 실행


```bash
uv run python fastapi_app.py
# http://localhost:8000 접속
```

# 메타 라마 워크숍 데모 - 고객 지원 에이전트

LangGraph와 로컬 LLM 분류를 활용한 실제 고객 지원 에이전트 시스템

## 주요 기능

- **로컬 LLM 분류**: Ollama를 사용한 고객 문의 분류
- **RAG 파이프라인**: 내부 지식 베이스 벡터 검색
- **웹 검색**: Tavily API를 통한 실시간 웹 검색
- **스마트 라우팅**: 검색 품질 기반 자동 전문가 에스컬레이션
- **인터랙티브 데모**: 인터럽트 기능이 있는 명령줄 인터페이스
- **웹 인터페이스**: FastAPI 기반 챗봇 UI
- **주피터 노트북**: 완전한 워크숍 자료

## 사전 요구사항

### Ollama 설치 (로컬 LLM 필수)

```bash
# macOS
brew install ollama

# 또는 다운로드: https://ollama.ai/download
```

필요한 모델 다운로드:
```bash
ollama pull llama3.2:3b
```

Ollama 서비스 시작:
```bash
ollama serve
```

### API 키

다음 API 키가 필요합니다:
- **Upstage API**: [여기서 발급](https://console.upstage.ai/)
- **Tavily API**: [여기서 발급](https://tavily.com/)

## 빠른 시작

### 1. 환경 설정

자세한 설정 방법은 `환경설정 가이드.md`를 참조하세요.

간단 설정:
```bash
# uv 설치 (Python 패키지 매니저)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 클론 및 설정
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

### 3. 실행 방법

**인터랙티브 명령줄:**
```bash
uv run python graph_build.py
```

**웹 인터페이스:**
```bash
uv run python fastapi_app.py
# http://localhost:8000 접속
```

**주피터 노트북:**
```bash
uv run jupyter notebook
# notebooks/Customer_Support_Agent_Complete.ipynb 열기
```

**LangGraph 개발 모드:**
```bash
uv run langgraph dev
# http://localhost:8123 접속
```

## 아키텍처

### 로컬 LLM 분류
- **모델**: Ollama를 통한 `llama3.2:3b`
- **목적**: 고객 문의를 카테고리별로 분류
- **폴백**: LLM 실패시 키워드 기반 분류
- **카테고리**: Technical, Billing, Account, General, Urgent

### RAG 파이프라인
- **임베딩**: Upstage 임베딩 모델
- **벡터 DB**: FAISS (인메모리)
- **검색**: 의미 유사도 검색
- **임계값**: 지식 베이스 결과 25% 유사도

### 웹 검색
- **제공자**: Tavily API
- **쿼리 최적화**: LLM 기반 쿼리 재작성
- **임계값**: 웹 결과 50% 관련성

### 스마트 라우팅
- **직접 답변**: 고품질 검색 결과 발견시
- **에스컬레이션**: 저품질 결과시 전문가 배정
- **전문가 배정**: LLM 분류 기반

## 프로젝트 구조

```
├── graph_build.py          # 메인 LangGraph 구현
├── fastapi_app.py          # 웹 인터페이스
├── 환경설정 가이드.md        # 한국어 설정 가이드
├── langgraph.json          # LangGraph 설정
├── data/
│   └── customer_data.csv   # 샘플 고객 데이터
├── notebooks/
│   └── Customer_Support_Agent_Complete.ipynb
└── static/                 # 웹 UI 자원
    ├── index.html
    ├── style.css
    └── script.js
```

## 설정

### 모델 설정
- **LLM**: `solar-pro2` (Upstage)
- **로컬 LLM**: `llama3.2:3b` (Ollama)
- **Temperature**: 0 (결정론적)

### 검색 임계값
- **지식 베이스**: 25% 유사도
- **웹 검색**: 50% 관련성
- **에스컬레이션**: 임계값 미만시

### 전문가 카테고리
- **Technical**: API, 시스템 이슈
- **Billing**: 결제, 구독
- **Account**: 로그인, 보안
- **General**: 일반 문의
- **Urgent**: 긴급 이슈

## 테스트

다음 예시 질문들을 시도해보세요:
- "비밀번호를 재설정하려면 어떻게 해야 하나요?"
- "API 연결에 문제가 있어요"
- "결제 정책이 궁금합니다"
- "시스템이 다운되었어요 - 긴급 도움이 필요합니다"

## 문제 해결

### Ollama 이슈
```bash
# Ollama 실행 확인
ollama list

# Ollama 재시작
ollama serve

# 모델 테스트
ollama run llama3.2:3b "안녕하세요"
```

### API 이슈
- `.env` 파일 존재 및 올바른 키 확인
- API 키 유효성 검증
- 네트워크 연결 확인

## 참고 자료

- [LangGraph 문서](https://langchain-ai.github.io/langgraph/)
- [Upstage API 문서](https://developers.upstage.ai/)
- [Tavily API 문서](https://docs.tavily.com/)
- [Ollama 문서](https://github.com/ollama/ollama)

## 라이선스

교육용 전용 - 메타 라마 워크숍 데모
# ADK-SELFRAG

Google ADK(Agent Development Kit)를 사용한 Self-RAG 구현

## 개요

이 프로젝트는 Google의 Agent Development Kit를 사용하여 Self-RAG(Retrieval Augmented Generation) 에이전트를 구현합니다. 이 에이전트는 문서를 처리하고, 관련 정보를 검색하며, 검색된 컨텍스트를 기반으로 응답을 생성할 수 있습니다.

## 프로젝트 구조

```
ADK-SELFRAG/
├── .env                   # 환경 변수 설정
├── .env.example           # 환경 변수 예시 파일
├── .gitignore             # Git 무시 파일
├── .python-version        # Python 버전 명세
├── LICENSE                # 프로젝트 라이센스
├── README.md              # 이 문서 파일
├── main.py                # 애플리케이션 메인 진입점
├── pdf_uploader.py        # PDF를 벡터 데이터베이스에 업로드하는 유틸리티
├── pyproject.toml         # 프로젝트 의존성 및 설정
├── uv.lock                # uv 패키지 매니저 락 파일
├── .venv/                 # 가상 환경 디렉토리
└── self-rag/              # 주요 패키지 디렉토리
    ├── __init__.py        # 패키지 초기화
    ├── __pycache__/       # Python 캐시 디렉토리
    ├── agent.py           # 필요한 모든 LlmAgent와 루트 에이전트 정의
    ├── custom_agent.py    # 사용자 정의 에이전트 확장
    ├── prompts.py         # 에이전트용 프롬프트 템플릿
    └── tools/             # 도구 디렉토리
        ├── __init__.py    # 도구 패키지 초기화
        ├── __pycache__/   # Python 캐시 디렉토리
        └── tools.py       # 도구 구현
```

## 필수 요구사항

- Python 3.9+
- Pinecone 계정과 API 키
- OpenAI API 키
- Google ADK 설정과 Google AI API 키
- uv 패키지 매니저 설치

## 설정

1. 리포지토리 복제:
   ```bash
   git clone https://github.com/jeyong-shin/adk-selfrag.git
   cd adk-selfrag
   ```

2. 가상 환경 생성 및 활성화:
   ```bash
   uv venv
   source .venv/bin/activate  # Windows에서는: .venv\Scripts\activate
   ```

3. 의존성 설치:
   ```bash
   uv pip install -e .
   ```

4. 환경 변수 설정:
   ```bash
   cp .env.example .env
   ```

5. `.env` 파일을 API 키로 수정:
   ```
   GOOGLE_GENAI_USE_VERTEXAI=FALSE
   GOOGLE_API_KEY=your_api_key
   OPENAI_API_KEY=your_api_key
   PINECONE_API_KEY=your_api_key
   PINECONE_INDEX_NAME=your_index_name
   PINECONE_NAMESPACE=your_namespace
   ```

## 사용법

### 웹 애플리케이션 실행

에이전트의 웹 인터페이스를 시작하려면:

```bash
uv run adk web
```

이렇게 하면 에이전트와 상호작용할 수 있는 로컬 웹 서버가 시작됩니다.

### 문서 업로드

에이전트를 실행하기 전에 Pinecone 인덱스에 문서를 업로드할 수 있습니다:

```bash
python pdf_uploader.py path/to/your/document.pdf namespace_name
```

이 유틸리티는 다음을 수행합니다:
1. PDF에서 텍스트 추출
2. 텍스트를 관리 가능한 청크로 분할
3. 각 청크에 대한 임베딩 생성
4. 임베딩을 Pinecone 인덱스에 업로드

## 에이전트 아키텍처

이 프로젝트는 여러 에이전트 구현을 포함합니다:

- `agent.py`: 핵심 LlmAgent 구현과 루트 에이전트를 포함
- `custom_agent.py`: 특정 사용 사례를 위한 사용자 정의 에이전트 확장
- `tools.py`: 에이전트가 작업을 수행하는 데 사용할 수 있는 도구 구현

## 라이센스

자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 문제 해결

문제가 발생하는 경우:

1. 모든 API 키가 `.env` 파일에 올바르게 설정되어 있는지 확인
2. Pinecone 인덱스가 올바르게 구성되어 있는지 확인
3. 호환되는 Python 버전을 사용하고 있는지 확인
4. uv 패키지 매니저가 설치되어 있고 올바르게 작동하는지 확인

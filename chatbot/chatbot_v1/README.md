# 한국어 특화 RAG 챗봇 시스템

## 프로젝트 개요

**한국어 특화 RAG(Retrieval-Augmented Generation) 챗봇 시스템**

문서 기반 질의응답과 일반 대화를 모두 지원하는 AI 챗봇으로, 사용자가 PDF 문서를 업로드하면 해당 문서의 내용을 기반으로 정확한 답변을 제공합니다.

---

## 기술 스택

### 백엔드 프레임워크
- **FastAPI**
  - 고성능 비동기 웹 프레임워크
  - RESTful API 설계
  - 자동 API 문서 생성 (Swagger UI)

### 프론트엔드
- **Streamlit**
  - Python 기반 웹 UI 프레임워크
  - 실시간 채팅 인터페이스
  - 파일 업로드 기능
  - 타이핑 애니메이션 효과

### AI/ML 핵심 기술

#### LLM (Large Language Model)
- **모델**: `Bllossom/llama-3.2-Korean-Bllossom-3B`
  - 한국어 특화 LLaMA 3.2 기반 모델
  - 30억 파라미터
  - Hugging Face 호스팅

#### 임베딩 모델
- **모델**: `intfloat/multilingual-e5-small`
  - 다국어 지원 임베딩 모델
  - 문서 벡터화에 최적화
  - 의미적 유사도 검색 지원

### LangChain 생태계
- **LangChain Core**
  - `InMemoryChatMessageHistory`: 대화 기록 관리
  - `HumanMessage`, `AIMessage`: 메시지 타입 정의

- **LangChain Components**
  - `ConversationalRetrievalChain`: RAG 체인 구성
  - `ConversationBufferMemory`: 대화 맥락 유지
  - `HuggingFacePipeline`: LLM 파이프라인 래퍼

### 벡터 데이터베이스
- **FAISS (Facebook AI Similarity Search)**
  - 고속 벡터 유사도 검색
  - MMR (Maximal Marginal Relevance) 검색 지원
  - 메모리 기반 벡터 저장소

### 문서 처리
- **PyMuPDF (fitz)**: PDF 파일 파싱
- **python-pptx**: PowerPoint 파일 처리
- **Docx2txt**: Word 문서 처리
- **RecursiveCharacterTextSplitter**: 문서 청킹

### 토크나이저
- **tiktoken**: OpenAI 토크나이저
  - 청크 크기 측정에 사용
  - cl100k_base 인코딩

### 하드웨어 가속
- **PyTorch + CUDA**
  - GPU 가속 추론
  - bfloat16 정밀도 최적화
- **Accelerate**: 모델 로딩 최적화

---

## 시스템 아키텍처

```
┌─────────────────┐
│   Streamlit UI  │ (ui.py)
│   - 채팅 인터페이스 │
│   - 파일 업로드    │
└────────┬────────┘
         │ HTTP Request
         │
┌────────▼────────┐
│  FastAPI Server │ (app.py)
│                 │
│  ┌─────────────┐│
│  │ /chat       ││ ← 일반 챗봇
│  ├─────────────┤│
│  │ /rag_chat   ││ ← RAG 챗봇
│  ├─────────────┤│
│  │ /upload_pdf ││ ← 파일 업로드
│  └─────────────┘│
└────────┬────────┘
         │
    ┌────┴─────┐
    │          │
┌───▼───┐  ┌──▼─────┐
│ChatBot│  │RaG_    │
│       │  │ChatBot │
└───┬───┘  └──┬─────┘
    │         │
    │    ┌────▼─────┐
    │    │  FAISS   │
    │    │ Vector DB│
    │    └──────────┘
    │
┌───▼──────────────┐
│ LLaMA 3.2 Korean │
│   (GPU/CUDA)     │
└──────────────────┘
```

---

## 시스템 동작 흐름

### 1. 일반 챗봇 모드 (`/chat`)

```
사용자 입력
    ↓
Streamlit UI (user_input)
    ↓
FastAPI POST /chat
    ↓
ChatBot.chat(user_input)
    ↓
LLaMA 3.2 모델 추론
    ↓
응답 후처리 (정규식으로 토큰 제거)
    ↓
InMemoryChatMessageHistory 저장
    ↓
JSON 응답 반환
    ↓
Streamlit에서 타이핑 애니메이션으로 표시
```

**코드 위치**:
- `app.py:128-153` - ChatBot 클래스
- `app.py:170-177` - `/chat` 엔드포인트

---

### 2. RAG 챗봇 모드 (`/rag_chat`)

#### ① 문서 업로드 단계:
```
사용자가 PDF 업로드
    ↓
Streamlit file_uploader
    ↓
FastAPI POST /upload_pdf
    ↓
PyMuPDF로 텍스트 추출
    ↓
document_store에 텍스트 저장
```

#### ② 질의응답 단계:
```
사용자 질문 입력
    ↓
FastAPI POST /rag_chat
    ↓
RecursiveCharacterTextSplitter로 텍스트 청킹
  ├─ chunk_size: 900 토큰
  └─ chunk_overlap: 100 토큰
    ↓
HuggingFaceEmbeddings로 벡터화
    ↓
FAISS 벡터 DB 구축
    ↓
ConversationalRetrievalChain 생성
  ├─ Retriever: MMR 검색
  ├─ Memory: ConversationBufferMemory
  └─ LLM: LLaMA 3.2
    ↓
관련 문서 검색 + 질문 조합
    ↓
LLM이 컨텍스트 기반 답변 생성
    ↓
응답 후처리 및 반환
```

**코드 위치**:
- `app.py:85-124` - RaG_ChatBot 클래스
- `app.py:181-192` - `/rag_chat` 엔드포인트
- `app.py:194-212` - `/upload_pdf` 엔드포인트

---

## UI 특징 및 사용자 경험

### Streamlit 인터페이스

1. **메인 화면**
   - 제목: "🤖 Iljoo Mini Chat-bot"
   - 청록색 테마 (rgb(0, 139, 139))
   - 화이트 배경

2. **사이드바**
   - PDF/DOCX 파일 업로드
   - 다중 파일 지원
   - Process 버튼으로 문서 처리

3. **채팅 인터페이스**
   - 사용자/봇 메시지 구분
   - 타이핑 애니메이션 (0.05초 간격)
   - 대화 기록 유지
   - "내용 지우기" 버튼 (GPU 메모리 해제 포함)

**코드 위치**: `ui.py:23-75`

---

## 핵심 파라미터 설정

### LLM 생성 파라미터
```python
max_new_tokens=100        # 최대 생성 토큰 수
temperature=0.5           # 창의성 (낮을수록 결정론적)
top_p=0.8                 # 누적 확률 샘플링
repetition_penalty=1.2    # 반복 억제
do_sample=True            # 샘플링 활성화
```
**위치**: `app.py:68-79`

### 텍스트 청킹 파라미터
```python
chunk_size=900           # 청크당 토큰 수
chunk_overlap=100        # 청크 간 중복 토큰
length_function=tiktoken_len  # 토큰 계산 함수
```
**위치**: `app.py:43-51`

### 벡터 검색 설정
```python
search_type='mmr'         # MMR 검색 알고리즘
device='cuda'             # GPU 가속
normalize_embeddings=True # 벡터 정규화
```
**위치**: `app.py:53-60`, `app.py:96`

---

## 주요 기술적 특징

### 1. 한국어 특화
- LLaMA 3.2 한국어 파인튜닝 모델 사용
- 한국어 질의응답에 최적화

### 2. RAG (Retrieval-Augmented Generation)
- 문서 기반 답변으로 환각(hallucination) 감소
- 실시간 문서 업로드 및 처리
- MMR 알고리즘으로 다양성 있는 검색 결과

### 3. 대화 맥락 유지
- InMemoryChatMessageHistory로 세션 관리
- ConversationBufferMemory로 이전 대화 참조
- 연속된 질문에 대한 맥락 이해

### 4. GPU 최적화
- CUDA 가속 지원
- bfloat16 정밀도로 메모리 효율 향상
- torch.cuda.empty_cache()로 메모리 관리

### 5. 응답 품질 향상
- 정규식 기반 후처리
- 불필요한 토큰 제거
- 깔끔한 답변 포맷팅

**코드 위치**:
- `app.py:111` - RAG 응답 후처리
- `app.py:143` - 일반 응답 후처리
- `ui.py:111-129` - UI 응답 정제

---

## API 엔드포인트

| 엔드포인트 | 메서드 | 기능 | 파라미터 |
|-----------|--------|------|----------|
| `/chat` | POST | 일반 대화 | `user_input: str` |
| `/rag_chat` | POST | 문서 기반 QA | `user_input: str` |
| `/upload_pdf` | POST | 문서 업로드 | `file: UploadFile` |

---

## 활용 시나리오

1. **기업 내부 문서 QA**
   - 매뉴얼, 정책 문서 업로드
   - 직원들의 빠른 정보 검색

2. **학습 보조**
   - 교재 PDF 업로드
   - 내용 요약 및 질의응답

3. **고객 지원**
   - 제품 설명서 기반 고객 문의 응답
   - 24/7 자동 응답 시스템

---

## 실행 방법

### 백엔드 실행
```bash
cd "F:\2.프로젝트\[AI-agent] LLM\chatbot\chatbot_v1"
uvicorn app:app --reload --port 8000
```

### 프론트엔드 실행
```bash
streamlit run ui.py
```

---

## 주요 특장점

✅ **기술 스택의 다양성**: FastAPI, Streamlit, LangChain, FAISS 등
✅ **AI/ML 이해도**: RAG 구현, 벡터 데이터베이스 활용
✅ **한국어 NLP**: 한국어 특화 모델 사용 경험
✅ **성능 최적화**: GPU 가속, 메모리 관리
✅ **사용자 경험**: 직관적인 UI, 타이핑 애니메이션
✅ **확장성**: 모듈화된 구조, RESTful API 설계

---

## 기술적 도전과제 및 해결방안

### 1. 토큰 제한 문제
- **문제**: LLM의 컨텍스트 윈도우 제한
- **해결**: RecursiveCharacterTextSplitter로 적절한 청크 크기 조정 (900 토큰)

### 2. 검색 품질
- **문제**: 단순 유사도 검색의 중복성
- **해결**: MMR 알고리즘으로 다양성과 관련성 균형

### 3. 메모리 관리
- **문제**: GPU 메모리 부족
- **해결**: bfloat16 정밀도 사용 및 명시적 캐시 해제

### 4. 응답 품질
- **문제**: LLM의 불필요한 토큰 생성
- **해결**: 정규식 기반 후처리 파이프라인 구축

---

## 라이선스

이 프로젝트는 개인 포트폴리오 목적으로 제작되었습니다.

---

## 문의

프로젝트에 대한 문의사항이 있으시면 연락 주시기 바랍니다.

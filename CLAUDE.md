# 노트메이트 (NoteMate) — 개발 가이드

> 이 문서는 VS Code의 Claude Code에게 전달하기 위한 개발 가이드입니다.
> **[확정]** 표시는 여러 차례 논의를 거쳐 정한 방향이고, **[잠정]** 표시는 구현하면서
> 편의성·효율성에 따라 바뀔 수 있는 부분입니다. Claude Code가 [잠정] 항목에서 더 나은
> 방법을 발견하면, 이유를 설명하고 대안을 제안해도 됩니다. [확정] 항목을 바꾸려면
> 먼저 이유를 설명하고 사용자 확인을 받아주세요.

## 한 줄 소개

노트메이트는 필기를 방해하지 않으면서도 곁에서 조용히 준비하는 AI 학습 동반자입니다.
**"AI는 항상 일하지만, 항상 말하지는 않는다"** — 이것이 이 서비스의 핵심 원칙입니다.

## MVP 범위 — 딱 이 3가지만 [확정]

프로젝트 가이드의 "기능 욕심 금지" 원칙에 따라, 아래 3개 기능만 완성도 있게 구현합니다.
나머지 아이디어(STT 대조, 유튜브 자막, 계층적 요약, 갭 분석, GraphDB 기반 개념 연결 등)는
**애드온으로 별도 모듈화**하되, MVP 완성 전에는 손대지 않습니다.

1. **실시간 필기 + 방해 없는 AI 힌트** (Tier 1/Tier 2 구조)
2. **개인화 RAG 되먹임 루프** (쓸수록 맞춤형이 되는 지식베이스)
3. **문서 업로드 → RAG 구축** (기존 필기/자료를 지식베이스의 기반으로)

## 기술 스펙

| 항목 | 선택 | 상태 | 비고 |
|---|---|---|---|
| 언어 | Python 3.11 | 확정 | |
| 에이전트 오케스트레이션 | LangGraph | 확정 | |
| RDBMS + Vector DB | **Supabase (PostgreSQL + pgvector 확장)** | 확정 | |
| 하이브리드 검색 | 벡터 유사도 + BM25/Kiwi 형태소분석 (EnsembleRetriever) | 확정 | |
| 문서 파싱 | Docling 기본 | 잠정 | 표/이미지가 많거나 Docling이 구조를 못 잡는 문서(예: 강사님 노션 내보내기)에서만 LlamaParse로 전환. Docling은 로컬 무료, LlamaParse는 외부 API 키·사용량 제한이 따르므로 기본값은 Docling 유지 |
| LLM | OpenAI API 기본 (`gpt-4.1-mini`), model-agnostic 구조 | 잠정 | LangChain의 `init_chat_model` 패턴으로 provider를 추상화 — `.env`의 모델 이름만 바꿔 Anthropic Claude 등으로 교체 가능하게 구현. 특정 provider를 코드에 하드코딩하지 말 것. Claude는 이후 비교/평가용으로 별도 사용 예정 |
| 임베딩 모델 | OpenAI `text-embedding-3-small` (1536차원) | 잠정 | Voyage AI 등으로 교체 시 `note_chunks.embedding` 컬럼 차원 재생성 필요 |
| 배포 | 로컬 개발은 `langgraph dev`, 이후 필요 시 Docker Compose만 | 확정 | 과한 배포 인프라 지양 |
| 프론트엔드 | 미확정 — 백엔드 검증 후 별도 논의 | 잠정 | 개발/테스트 단계는 LangGraph Studio Chat UI로 대체 |

**명시적으로 빼는 것 (MVP 범위 아님):**
- GraphDB(Neo4j 등) — 개념 간 관계/학습경로 추천은 향후 확장 아이디어로만 기록, 지금 구현하지 않음
- InMemory Vector Store를 메인 DB와 **병렬 동기화되는 상시 운영 계층**으로 쓰는 구조는 제외 — 대신 세션 범위로 좁혀서 구현 (아래 "세션 내 작업기억" 참고)
- Kubernetes — PoC 단계에서 불필요

## 데이터 흐름

### 1. 표준 필기 조각(Standard Note Chunk) — 모든 입력의 공통 형식

서로 다른 소스(txt 필기, 종이 스캔 OCR, 강사 노션 자료, 이후 붙일 유튜브 자막/STT)를
각각의 **어댑터**가 아래 공통 형식으로 변환합니다. 이후 파이프라인은 이 형식만 다룹니다.

```python
class NoteChunk(BaseModel):
    content: str
    source: Literal["personal", "base", "stt_reference"]  # 필기 / 개론자료 / 참고소스
    medium: Literal["txt", "paper_ocr", "notion", "video_caption"]
    course_id: str | None          # 주차/토픽 단위로 운용 (예: "langgraph_week")
    timestamp: datetime | None      # 모르면 None — 임의로 채우지 않음
    origin_ref: str                 # 원본 파일/줄 위치 (원문 보기용)
```

### 2. 전처리
- 문서(txt/PDF/노션): Docling으로 파싱 → 청킹 → 위 표준 형식으로 변환. Docling이 구조를 못 잡는 문서만 LlamaParse로 개별 전환 (위 기술 스펙 표 참고)
- 종이 필기: OCR 후 동일 형식으로 변환, `medium: paper_ocr`로 표시(정확도 낮을 수 있음을 구분)
- 실시간 필기(추후 프론트엔드 연결 시): 타이핑 멈춤 감지 등으로 조건부 트리거

### 3. 저장소 — Supabase 단일 DB
- 필기 원본, 메타데이터(course_id, source, medium, timestamp), 벡터 임베딩을 **하나의 테이블/DB에** 저장
- 파인콘+Postgres 혼합 대신 pgvector 통합을 택한 이유: 트랜잭션 정합성, 복합 SQL 쿼리 자유도, 키 관리 단순화 (상세 근거는 별도 논의 기록 참고)

### 4. 확신도(confidence) 기반 Tier 1/Tier 2 라우팅 — 이 서비스의 핵심 로직

두 가지 서로 다른 신호를 결합합니다. 하나만 보면 오판합니다.

- **신호 A: 지원 자료 충분도** — 벡터 검색 시 나오는 유사도 점수 (정량, 거의 공짜)
- **신호 B: 필기 완결도** — LLM이 정리 노드에서 구조화 출력으로 함께 판단 (정성)

```python
class NoteAssessment(BaseModel):
    is_question: bool
    is_abbreviated: bool
    is_new_concept: bool
```

```python
def route(state):
    if state["is_question"]:
        return "tier2"
    if state["top_score"] < LOW_THRESHOLD:      # [잠정] 값은 실제 데이터로 튜닝
        return "tier2"
    if state["is_abbreviated"] and state["top_score"] < HIGH_THRESHOLD:  # [잠정]
        return "tier2"
    return "store_only"
```

- `LOW_THRESHOLD`, `HIGH_THRESHOLD`는 지금 확정할 수 없음 — 실제 필기 데이터로 재생(replay) 테스트하며 결정 [잠정]
- **Tier 1** (항상 조용히 동작): 위 두 신호 계산 + 얕은 검색, 결과를 "앵커 카드"에 저장만 하고 표시하지 않음
- **Tier 2** (확신도 낮거나 명시적 질문일 때만): 딥 에이전트가 단기/장기 기억 전체 검색, 필요시 원본 드릴다운까지 수행 후 카드 보강
- 사용자에게 보이는 "점(신호) 표시"는 Tier 2가 실제로 개입한 카드에만 붙임 — 모든 줄에 붙지 않도록 주의

### 5. 개인화 되먹임 루프
- 정리된 필기가 세션 종료/유휴 시점에 배치로 지식베이스에 upsert
- `course_id`, `source`, `medium` 메타데이터로 검색 시 필터링

### 6. 세션 내 작업기억 (InMemory VectorStore) — 좁힌 역할로 활용

메인 DB(pgvector)와는 별개로, **현재 세션 안에서만** 존재하는 InMemory VectorStore를 하나 둡니다.
메인 DB는 "이미 정식으로 쌓인 지식"을 담당하고, 이 작업기억은 "오늘 세션에서 방금 다룬 것"만 담당합니다 — 역할이 겹치지 않습니다.

- **범위**: 이번 세션의 필기만. 세션 시작 시 비어서 시작, 세션 종료 시 폐기(정식 저장은 되먹임 루프가 별도로 담당)
- **용도 1 — 중복 힌트 방지**: 방금 필기가 몇 분 전 세션 내에서 이미 다룬 내용과 비슷하면, 굳이 Tier 2를 다시 돌리지 않음
- **용도 2 — 긴 세션의 맥락 압축**: 세션이 길어져 전체를 프롬프트에 넣을 수 없을 때, 방금 필기와 의미적으로 가장 관련 있는 세션 내 이전 필기만 뽑아 컨텍스트로 사용
- [잠정] 중복 판정 임계값도 `LOW_THRESHOLD`/`HIGH_THRESHOLD`처럼 실제 데이터로 튜닝 필요

### 7. 판단 로그 — 별도 로그 테이블 대신 필기 조각 테이블에 컬럼으로 기록

InMemory 작업기억이나 Tier 1의 판단 "결과"는 기록하되, 벡터 값 자체를 영구 저장하지는 않습니다
(세션 종료 시 폐기된다는 정의와 모순되므로). 대신 표준 필기 조각(`NoteChunk`) 테이블에
판단 결과 컬럼을 추가해, 필기 한 줄과 판단 근거가 한 행에 남도록 합니다.

```python
class NoteChunk(BaseModel):
    # ... 위 표준 형식 필드 ...
    top_score: float | None                  # 신호 A: 메인 DB 검색 최고 유사도
    is_question: bool | None                 # 신호 B (구조화 출력)
    is_abbreviated: bool | None
    is_new_concept: bool | None
    session_duplicate_score: float | None    # 세션 내 작업기억 중복 검사 결과
    routing_decision: Literal["tier2", "store_only"] | None
```

이렇게 하면 별도 로그 테이블(User_Log/Vector_Log/Search_Log 등) 없이도 `WHERE routing_decision = 'tier2'` 같은
SQL 한 줄로 "전체 필기 중 Tier 2 개입 비율", "임계값 조정 전후 비교" 같은 정량 결과를 뽑을 수 있습니다.
발표자료의 "테스트 방법 및 결과", "어려웠던 점" 섹션 근거로 바로 활용 가능합니다.

## 백엔드 검증 방식 — 프론트엔드 없이 테스트하기

프론트엔드가 아직 없으므로, "필기 입력 → 개인화 반영"을 아래 방식으로 검증합니다.

1. `ingest_note(text, timestamp, course_id, ...)` 함수를 백엔드 API의 첫 진입점으로 설계
2. 사용자의 실제 3개월치 txt 필기를 시간순으로 파싱해 `ingest_note()`에 순차 투입하는 재생(replay) 스크립트 작성
3. `langgraph dev` + LangGraph Studio Chat UI를 디버깅/테스트 플랫폼으로 사용 — State 변화, Tier1/2 라우팅 판단을 Studio에서 직접 확인
4. 같은 질문을 필기 투입량이 다른 시점(1주차만 vs 8주차까지)에 던져 답변이 개인화되는지 비교
5. LangSmith Tracing으로 각 청크의 라우팅 판단 근거 로그 확인

이 재생 스크립트의 `ingest_note()`가 나중에 프론트엔드가 호출할 실제 API가 됩니다 — 임시 테스트 코드가 아닙니다.

## 개발 순서 제안

1. 프로젝트 뼈대 (`langgraph.json`, `src/app.py, state.py, nodes.py`, `.env` 템플릿)
2. Supabase 스키마 설계 + pgvector 확장 설정
3. 표준 필기 조각 어댑터 (txt부터 시작, OCR/노션은 이후)
4. `ingest_note()` + 재생 스크립트
5. 정리 노드 + Tier 1 (확신도 계산 포함)
6. Tier 2 딥 에이전트 + 조건부 라우팅
7. 개인화 되먹임 루프
8. (MVP 완성 후) 프론트엔드 별도 논의

## 주의사항

- API 키는 `.env`에만, 커밋 금지
- 패키지 버전 충돌 시 임의로 우회하지 말고 먼저 보고할 것
- [잠정] 표시된 항목은 구현 중 더 나은 대안이 보이면 이유와 함께 제안 가능
- [확정] 항목을 변경해야 할 기술적 이유가 생기면, 먼저 설명하고 확인을 받을 것

## 변경 이력

- 2026-09-29: LLM 기본값 [잠정] Anthropic Claude → **OpenAI API (`gpt-4.1-mini`)** — 사용자가 발급받은 키로 gpt-4.1-mini 사용 예정, Claude는 이후 비교용으로 별도 사용
- 2026-09-29: 임베딩 모델 [잠정] 미정 → **OpenAI `text-embedding-3-small` (1536차원)**로 확정 — LLM을 OpenAI로 정한 김에 기본값 채택
- 2026-09-29: Supabase DB 연결 방식 — `supabase-py`(REST) 대신 **Postgres 직접 연결**(`psycopg` + `POSTGRES_URI`) 채택. 이유: 커스텀 판단 로그 컬럼 포함한 복합 SQL 쿼리 자유도 확보, REST 레이어 불필요. Connection은 Supavisor **Session pooler**(5432) 사용
- 2026-09-29: `note_chunks` 테이블 생성 (Supabase 프로젝트 `nqeqzvcutqmrtmaqobbo`, `ap-southeast-2`) — NoteChunk 표준 필드 + 판단 로그 컬럼 + `embedding VECTOR(1536)`, HNSW 인덱스 적용. SQL 예약어 충돌 방지 위해 Python 모델의 `timestamp` 필드는 DB 컬럼명 `note_timestamp`로 매핑
- 2026-09-29: 의존성 추가 — `psycopg[binary,pool]`, `pgvector` (Python 패키지)
- 2026-09-29: `NoteMateState`(`src/state.py`) 설계 — `messages` 필드 미포함으로 결정. 서비스 진입점이 채팅이 아니라 `ingest_note()` 함수 호출이라, LangGraph Studio Chat 탭 대신 Input 폼 + Trace 뷰로 테스트. 테스트 중 불편하면 나중에 얇은 변환 노드로 추가 가능
- 2026-09-29: `decide_routing` 노드 추가 — 조건부 엣지 함수(`routers.route`)는 State에 값을 못 쓴다는 걸 발견, 판단 로직(`compute_routing_decision`)과 State 기록(`decide_routing` 노드)을 분리
- 2026-09-29: `assess_note` 실제 구현 — `src/llm.py`(채팅/임베딩 모델), `src/db.py`(Supabase 쿼리 헬퍼) 신규 추가. 비교할 기존 필기가 없을 때(top_score=None) "확신도 낮음"과 동일하게 0.0으로 처리 -> 자연히 tier2로 라우팅되도록 설계
- 2026-09-29: pgvector 쿼리 이슈 발견/수정 — psycopg에 Python list를 그냥 넘기면 `double precision[]`로 직렬화돼 `<=>` 연산자가 타입 불일치로 실패함. `register_vector` 어댑터 등록 대신 SQL에서 `%s::vector`로 명시적 캐스팅하는 방식으로 해결 (`src/db.py`)
- 2026-09-29: `persist_note` 공통 노드 추가 — Tier1(`store_only`)/Tier2(`tier2_deep_agent`) 양쪽 경로 모두 끝에 이 노드를 거쳐 `note_chunks`에 저장. 가이드 원문의 "세션 종료/유휴 시점 배치 upsert"는 [잠정] 즉시 저장으로 단순화 (재생 스크립트 단계라 세션 개념이 아직 없음). 개인화 되먹임 루프 실제 동작 확인됨(유사 필기 재입력 시 top_score 0.0 -> 0.86로 상승)
- 2026-09-29: `tier2_deep_agent` 구현 — `note_chunks`에서 top-k(5개) 유사 필기 검색 -> LLM이 그 근거로 짧은 힌트 생성(`tier2_result`). 단기 기억(세션 내 InMemory VectorStore) 검색과 원본 드릴다운은 아직 TODO — 세션 개념 자체가 없어서 보류
- 2026-09-29: `ingest_note()` 구현 (`src/api.py` 신규) — 가이드에서 정의한 백엔드 API 진짜 진입점. `datetime` 인자를 받아 State가 기대하는 ISO 문자열로 변환 후 `graph.invoke()` 호출. 재생 스크립트/프론트엔드가 이 함수 하나만 쓰면 됨
- 2026-09-30: txt 어댑터(`src/adapters.py`) + 재생 스크립트(`src/replay.py`) 구현. txt 파일 형식(빈 줄로 필기 구분, `[YYYY-MM-DD HH:MM]` 선택적 타임스탬프)은 실제 사용자 파일을 아직 못 봐서 [잠정]으로 정함 — 나중에 실제 파일 형식 보면 조정 필요. 합성 테스트 데이터(`src/sample_notes.txt`, 9개 필기)로 재생 실행 -> 전부 tier2로 라우팅됨(9개가 서로 다른 주제라 유사도가 낮아서, 임계값 튜닝이 왜 필요한지 보여주는 사례). store_only 분기는 별도로 기존 필기와 거의 동일한 문장(top_score=1.0, 질문/축약 아님)을 넣어 정상 작동 확인
- 2026-09-30: 임계값 1차 검증 (합성 데이터, `src/sample_notes_tuning.txt`) — 같은 주제를 3주 간격으로 다르게 표현한 12개 필기로 재생 테스트. top_score가 뚜렷한 군집을 보임: 신규 개념 0.00~0.36 / 패러프레이즈 복습 0.58~0.76 / 거의 동일한 재입력 1.00. 질문형 문장은 점수 무관하게 항상 tier2로 정상 오버라이드됨. 현재 `LOW_THRESHOLD=0.75`, `HIGH_THRESHOLD=0.85`가 패러프레이즈 군집과 완전 중복 군집 사이 경계에 합리적으로 위치해 값 변경 없이 유지하기로 함 -- [잠정] 실제 필기 데이터로 재검증 필요, 분포가 다를 수 있음
- 2026-09-30: `ingest_file()` 추가 (`src/api.py`) — 사용자가 "업로드 버튼 하나로 txt든 PDF든 알아서 RAG로 변환"되길 원한다는 지적 반영. 확장자로 어댑터를 자동 선택해 `ingest_note()`로 넘기는 관문 함수. 지금은 `.txt`만 등록돼 있고, `_FILE_ADAPTERS` 딕셔너리에 한 줄만 추가하면 새 형식(PDF 등) 지원 가능한 구조로 설계. `replay.py`도 이 함수를 쓰도록 리팩터링(직접 파싱하던 코드 제거) -- Docling 어댑터 완성되면 재생 스크립트 수정 없이 그대로 PDF 재생 가능해짐. 참고: 현재 `medium` Literal(txt/paper_ocr/notion/video_caption)에 PDF에 해당하는 값이 없음 -- Docling 어댑터 만들 때 결정 필요 (신규 값 추가 or 기존 값 중 하나로 매핑)
- 2026-09-30: Docling 어댑터(`parse_document_notes`, `src/adapters.py`) 구현 — PDF/노션 내보내기(html/md) 공통 처리. `HybridChunker.contextualize()`로 각 청크 앞에 소속 헤딩 경로를 붙여 임베딩 품질 향상. 문서엔 작성 시각이 없어 timestamp는 항상 None. `note_chunks.medium` CHECK 제약에 `pdf` 값 추가(Supabase에 직접 ALTER), `_FILE_ADAPTERS`에 `.pdf`->`pdf`, `.html`/`.md`->`notion`(노션 내보내기 가정, [잠정]) 등록. `docling`은 무거운 의존성(torch 등)이라 함수 내부 지연 import
- 2026-09-30: **그래프 구조 변경** -- START 직후 `route_by_source`로 첫 갈림길 추가. `source="personal"`만 기존 Tier1/2 전체 파이프라인을 타고, 업로드된 참고자료(`base`/`stt_reference`)는 `embed_and_store` 노드(임베딩만 생성) -> `persist_note`로 바로 가는 경로 신설. 이유: 참고자료엔 "질문/축약 여부 판단"이나 "힌트 생성"이 의미가 없고(누구에게 줄 힌트인지 불분명), LLM 호출 2번(판단+힌트)이 청크마다 낭비되며, Tier2 개입 비율 같은 통계에 필기가 아닌 자료가 섞이는 문제가 있었음. 회귀 테스트로 두 경로 모두 정상 확인(참고자료: 판단/라우팅/힌트 필드 전부 None / 개인 필기: 기존과 동일하게 동작)
- 2026-09-30: `data/{personal,base,stt_reference}/` 업로드 폴더 컨벤션 추가 -- 폴더명이 곧 `source` 값. `data/`는 gitignore(실제 필기/저작권 있는 자료라 커밋 안 함), `.gitkeep`으로 구조만 추적. `src/ingest_folder.py`로 폴더 스캔 후 일괄 `ingest_file()` 실행
- 2026-09-30: **실제 필기 파일로 어댑터 재검증 (`parse_txt_notes` 교체)** -- 실제 사용자 파일은 "파일 하나 = 하루치 필기 전체, 조각별 타임스탬프 없음, 파일명 앞 8자리(YYYYMMDD)가 그날 날짜"인 것으로 확인됨. 기존에 가정했던 "조각마다 [YYYY-MM-DD HH:MM] 인라인 표시" 형식은 틀렸던 것으로 판명 -- 파일명에서 날짜를 추출해 그 날짜를 파일 내 모든 청크에 공통 적용하는 방식으로 교체. 실제 하루치 파일(23개 청크, 웹기초/n8n/텔레그램봇/JSON스키마 등 새로운 내용 위주)로 재생 테스트: 전부 tier2로 라우팅됨(같은 날 반복되는 내용이 없어 당연한 결과), 힌트 품질은 `"<<<n8n실습>>>"` 같은 짧고 애매한 조각에도 대체로 실질적이고 정확함. 유일한 약점: 날짜/장소/점심메뉴 같은 헤더성 메타정보 조각에도 무의미한 힌트가 생성됨 -- 해롭진 않지만 낭비. [잠정] 나중에 이런 헤더 패턴을 걸러내는 필터 추가 고려 가능, 지금은 MVP 우선으로 보류

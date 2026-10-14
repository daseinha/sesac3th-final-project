"""LangGraph 파이프라인이 공유하는 State 정의.

CLAUDE.md의 표준 필기 조각(NoteChunk) 스키마 + 판단 로그 컬럼을 그대로 반영한다.
messages 필드는 의도적으로 두지 않음 — 실제 서비스 진입점은 채팅이 아니라
ingest_note() 함수 호출이라서 (CLAUDE.md 참고, 2026-09-29 논의).
"""

from typing import Literal, TypedDict


class NoteAssessment(TypedDict):
    """정리 노드가 LLM 구조화 출력으로 채우는 신호 B (필기 완결도 판단)."""

    is_question: bool
    is_abbreviated: bool
    is_new_concept: bool


class NoteMateState(TypedDict):
    # --- 입력 (ingest_note()가 채움) ---
    content: str
    source: Literal["personal", "base", "stt_reference"]
    medium: Literal["txt", "paper_ocr", "notion", "video_caption"]
    course_id: str | None
    note_timestamp: str | None  # ISO 문자열. DB 컬럼명(note_timestamp)과 통일
    origin_ref: str
    session_id: str | None  # 호출하는 쪽이 지정. 세션 경계 판단은 백엔드가 안 함
    client_line_id: str | None  # 프론트엔드의 필기 "줄" 식별자 -- 같은 줄을 수정해 재제출해도
    # 같은 값이 유지됨. 파이프라인 판단에는 전혀 안 쓰이고 persist_note가 그대로 저장만 함 --
    # get_all_notes()가 "전체 복원 화면"에서 한 줄의 여러 수정본을 중복 표시하지 않고
    # 최신 것만 고르게 해주는 키(2026-10-02, 새로고침 시 수정 내역이 전부 중복으로
    # 쌓여 보이던 버그 수정).
    upload_label: str | None  # 2026-10-12: 자료 업로드 UI에서 사용자가 고르거나 새로
    # 만든 "분류명"(과목 선택 드롭다운과 같은 UX). 판단/라우팅엔 전혀 안 쓰이는 순수
    # 통과 필드 -- client_line_id/inserted_out_of_place와 같은 패턴. 업로드 경로에서만
    # 채워지고, 필기 입력 경로는 항상 None.
    is_ask: bool  # 2026-10-12: "바로 물어보기"로 들어온 요청인가(서버가 origin_prefix
    # == "ask"로 자동 판별해서 넘김). tier2_deep_agent가 이 값 + 근거 부족을 같이
    # 봐서 웹검색(Tavily) 호출 여부를 결정 -- 조용한 힌트 경로는 절대 안 씀
    # ([[deferred-web-search-addon]] 스코프 그대로: 바로 물어보기 전용).
    allow_general_knowledge: bool  # 2026-10-12: Tier2가 참고할 만한 근거(강의자료/
    # 내 필기)를 못 찾았을 때도 LLM 자기 지식만으로 힌트를 생성할지 여부. False면
    # 근거가 없는 경우 힌트 생성 자체를 건너뛴다(tier2_deep_agent 참고) -- "힌트
    # 남발" 우려 완화 + 근거 유무별 비교테스트용. 기본값은 True(기존 동작 그대로).
    dry_run: bool  # 2026-10-10: test:: 태그 입력을 "백엔드 호출 자체 스킵"에서
    # "기존 DB 근거로 실제 검색/힌트 생성은 돌리되 저장만 스킵"으로 바꾸기 위해
    # 추가. True면 persist_note가 DB insert와 세션 작업기억 등록을 건너뛴다 --
    # 그 앞 노드들(assess_note, tier2_deep_agent 등)은 전부 읽기만 하므로 안 건드림.
    inserted_out_of_place: bool | None  # 2026-10-10: 사용자가 타임라인 중 과거
    # 세션/날짜 블록으로 스크롤해서 그 자리에 새로 입력한 경우 True. client_line_id와
    # 같은 성격의 순수 통과 필드 -- 판단/라우팅엔 전혀 안 쓰이고 persist_note가
    # 그대로 저장만 함. 새로고침하면 이 줄은 (아직 미구현인 "소속 재배치"
    # 기능 전까지는) 여전히 오늘/활성 세션 자리로 재배치되지만, 이 플래그로
    # "원래 다른 위치에서 입력됐다"는 걸 화면에 다른 글자색으로만 표시해준다
    # ([[deferred-historical-note-insertion]] 참고 -- 제대로 된 기능은 차후과제).

    # --- 처리 중간 산출물 ---
    embedding: list[float] | None
    top_score: float | None  # 신호 A: 벡터 검색 최고 유사도
    assessment: NoteAssessment | None  # 신호 B: LLM 구조화 출력
    session_duplicate_score: float | None  # 세션 내 작업기억 중복 검사 결과
    session_duplicate_elapsed_seconds: float | None  # 그 중복과의 경과 시간(초) --
    # 라우팅 판단(15분 예외)에만 쓰는 임시 필드. DB 컬럼 없음 -- persist_note가
    # 저장하지 않음(2026-10-05, session_duplicate_score만 판단 로그로 남기기로 함).
    top_match_feedback_value: str | None  # 가장 비슷한 과거 필기의 피드백(+/-/✓) --
    top_match_feedback_days_ago: float | None  # 그 피드백이 며칠 전인지(2026-10-08,
    # 힌트 피드백 반영 Phase 2: ✓ 억제). session_duplicate_score와 같은 패턴으로
    # DB 컬럼에도 남긴다(persist_note) -- "억제가 실제로 몇 번 발동했는지"를
    # 나중에 쿼리로 뽑을 수 있게(`routing_decision='store_only' AND
    # top_match_feedback_value='✓'`이면 그때 발동한 것).

    # --- 라우팅 결과 ---
    routing_decision: Literal["tier2", "store_only"] | None
    tier2_result: str | None  # Tier2 딥 에이전트가 보강한 카드 내용
    tier2_sources: str | None  # 그 힌트의 근거 요약(2026-10-09) -- "내 필기 2건 · 강의자료
    # 1건" 또는 검색 결과가 없으면 "일반 지식(LLM 사전학습) 기반". 강사 피드백 +
    # 사용자 질문("이거 그냥 챗봇과 뭐가 다르냐") 반영 -- tier2_result만으론 그 답이
    # 어디서 왔는지 사용자가 알 길이 없었음. tier2_result가 있을 때만 의미 있음.
    hint_category: str | None  # 힌트 유형(2026-10-08) -- question_answer/new_concept/
    # abbreviated_reminder/general_nudge 중 하나, tier2_result가 있을 때만 의미 있음.
    # 추가 LLM 호출 없이 이미 계산된 assessment 신호 조합으로 결정론적으로 매김.
    # 나중에(Phase 4) "이 유형의 피드백 합계가 좋으면 더 자주" 반영할 때 묶는 축.

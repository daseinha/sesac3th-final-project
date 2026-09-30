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

    # --- 처리 중간 산출물 ---
    embedding: list[float] | None
    top_score: float | None  # 신호 A: 벡터 검색 최고 유사도
    assessment: NoteAssessment | None  # 신호 B: LLM 구조화 출력
    session_duplicate_score: float | None  # 세션 내 작업기억 중복 검사 결과

    # --- 라우팅 결과 ---
    routing_decision: Literal["tier2", "store_only"] | None
    tier2_result: str | None  # Tier2 딥 에이전트가 보강한 카드 내용

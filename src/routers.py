"""라우팅 로직 -- CLAUDE.md 4번 섹션의 Tier1/2 라우팅 규칙 그대로 구현.

LOW_THRESHOLD, HIGH_THRESHOLD는 [잠정] -- 실제 필기 데이터로 재생(replay)
테스트하며 튜닝 예정.

주의: LangGraph의 조건부 엣지 함수(add_conditional_edges에 넘기는 함수)는
다음 노드 이름만 반환할 뿐, State에 값을 써넣지 못한다. 그래서 판단 로직은
`compute_routing_decision()`에 두고, 실제로 State에 기록하는 건
nodes.py의 decide_routing 노드가 담당한다. `route()`는 그 노드가 이미
State에 저장해둔 값을 그대로 읽어서 다음 노드만 골라준다.
"""

from typing import Literal

from src.state import NoteMateState

LOW_THRESHOLD = 0.75  # [잠정] 재생 테스트로 튜닝 예정
HIGH_THRESHOLD = 0.85  # [잠정] 재생 테스트로 튜닝 예정
SESSION_DUPLICATE_THRESHOLD = 0.85  # [잠정] 세션 내 중복 판정 기준 -- 실데이터 튜닝 필요


def compute_routing_decision(state: NoteMateState) -> Literal["tier2", "store_only"]:
    assessment = state["assessment"]
    top_score = state["top_score"]

    # 세션 내에서 방금 다룬 내용과 거의 같으면, 질문이든 뭐든 상관없이 중복
    # 힌트를 만들지 않는다 (CLAUDE.md 6번 섹션 "용도1" -- 이미 방금 도와줬으므로).
    session_dup = state.get("session_duplicate_score")
    if session_dup is not None and session_dup >= SESSION_DUPLICATE_THRESHOLD:
        return "store_only"

    if assessment and assessment["is_question"]:
        return "tier2"
    # is_new_concept은 계산만 되고 라우팅엔 한 번도 안 쓰이고 있었음(2026-10-02,
    # 6/8일차 리뷰 중 발견 -- CLAUDE.md "라우팅 기준 보강" 변경 이력 참고).
    # LLM이 명시적으로 "새 개념"이라고 판단했다면 top_score/is_abbreviated와
    # 무관하게 보강하는 게 맞다고 판단해 여기 추가. session_dup 체크는 이
    # 조건보다 먼저 와서(위) -- 세션 내에서 방금 다룬 내용이면 "새 개념"
    # 판단이어도 중복 힌트를 또 만들지 않음(일관성 유지).
    if assessment and assessment["is_new_concept"]:
        return "tier2"
    if top_score is not None and top_score < LOW_THRESHOLD:
        return "tier2"
    if (
        assessment
        and assessment["is_abbreviated"]
        and top_score is not None
        and top_score < HIGH_THRESHOLD
    ):
        return "tier2"
    return "store_only"


def route(state: NoteMateState) -> Literal["tier2", "store_only"]:
    """조건부 엣지 함수 -- decide_routing 노드가 이미 State에 저장해둔
    routing_decision을 그대로 반환한다 (판단 로직 중복 방지)."""
    return state["routing_decision"]


def route_by_source(state: NoteMateState) -> Literal["full_pipeline", "reference_only"]:
    """그래프 진입 시점의 첫 갈림길 -- source에 따라 완전히 다른 경로로 보낸다.

    "personal"(사용자가 지금 쓰는 필기)만 Tier1/2 전체 파이프라인(정리 노드 ->
    라우팅 -> 힌트 생성)을 탄다. "base"/"stt_reference"(업로드된 참고자료)는
    힌트/넛지가 필요 없는 자료라서 -- 임베딩만 만들어 바로 저장하는
    가벼운 경로로 보낸다 (LLM 호출 2번어치 절약 + 판단 로그 통계 오염 방지).
    """
    if state["source"] == "personal":
        return "full_pipeline"
    return "reference_only"

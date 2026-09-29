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


def compute_routing_decision(state: NoteMateState) -> Literal["tier2", "store_only"]:
    assessment = state["assessment"]
    top_score = state["top_score"]

    if assessment and assessment["is_question"]:
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

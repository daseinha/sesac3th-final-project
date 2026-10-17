"""라우팅 로직 -- CLAUDE.md 4번 섹션의 Tier1/2 라우팅 규칙 그대로 구현.

LOW_THRESHOLD, HIGH_THRESHOLD는 [잠정] -- 실제 필기 데이터로 재생(replay)
테스트하며 튜닝 예정.

주의: LangGraph의 조건부 엣지 함수(add_conditional_edges에 넘기는 함수)는
다음 노드 이름만 반환할 뿐, State에 값을 써넣지 못한다. 그래서 판단 로직은
`compute_routing_decision()`에 두고, 실제로 State에 기록하는 건
nodes.py의 decide_routing 노드가 담당한다. `route()`는 그 노드가 이미
State에 저장해둔 값을 그대로 읽어서 다음 노드만 골라준다.
"""

import os
from typing import Literal

from src.state import NoteMateState

LOW_THRESHOLD = 0.75  # [잠정] 재생 테스트로 튜닝 예정
HIGH_THRESHOLD = 0.85  # [잠정] 재생 테스트로 튜닝 예정
SESSION_DUPLICATE_THRESHOLD = 0.85  # [잠정] 세션 내 중복 판정 기준 -- 실데이터 튜닝 필요
# 세션 내 중복이어도 이만큼 시간이 지났으면 억제를 풀어준다(2026-10-05, 리뷰
# 피드백 "같은 질문을 한참 뒤 다시 적었으면 중복이 아니라 재질문일 수 있다"
# 반영). .env로 덮어쓸 수 있게 해서 발표 시연 때는 짧게 줄일 수 있음.
SESSION_DUP_REALLOW_SECONDS = int(os.environ.get("SESSION_DUP_REALLOW_SECONDS", str(15 * 60)))
# 힌트 피드백 반영 Phase 2(✓ 억제, 2026-10-08, HINT_FEEDBACK_SPEC.md 열린 질문 2번
# "✓ 억제 기간을 몇 일로 시작할지"). 근거 데이터가 거의 없어 임의의 초기값으로
# 시작 -- 실사용 데이터가 쌓이면 재검토 필요. .env로 덮어쓸 수 있게 해서 발표
# 시연 때는 짧게 줄일 수 있음(다른 임계값들과 같은 패턴).
FEEDBACK_SUPPRESS_DAYS = float(os.environ.get("FEEDBACK_SUPPRESS_DAYS", "7"))
# 힌트 피드백 +/- 반영 (2026-10-17, 교육장에서 합의한 2단계 설계).
# tier2_deep_agent가 검색된 비슷한 과거 힌트 중 `-` 피드백 개수를 세어, 이
# 임계치를 넘으면(= 몇 번 다르게 시도해봤는데도 계속 안 맞았다는 뜻) 힌트
# 생성을 억제한다(질문이면 예외 -- 질문엔 무조건 답한다는 기존 원칙과 같은
# 결). 임계치 미만이면 억제 대신 "다르게 설명해보라"는 지시만 추가해서 먼저
# 개선을 시도한다 -- `✓`와 다르게 `-` 누적은 "사용자가 이미 안다"가 아니라
# "AI가 그 내용을 계속 잘 못 설명했다"일 수 있어서, 바로 침묵하면 가장
# 나쁜 결과(도움이 필요한데 조용히 포기)가 될 수 있다는 판단(사용자 지적:
# "피드백이 부정적이더라도, 그에 대한 대답은 긍정적으로 유도하자").
NEGATIVE_FEEDBACK_SUPPRESS_THRESHOLD = int(os.environ.get("NEGATIVE_FEEDBACK_SUPPRESS_THRESHOLD", "2"))


def compute_routing_decision(state: NoteMateState) -> Literal["tier2", "store_only"]:
    assessment = state["assessment"]
    top_score = state["top_score"]

    # 질문은 사용자가 명시적으로 "알고 싶다"는 의도를 표시한 것이므로, 세션
    # 중복 억제보다 우선순위가 높다(2026-10-14, 사용자 결정으로 순서 변경 --
    # 원래는 세션중복 체크가 더 먼저였고 "질문이든 뭐든 상관없이" 억제했는데,
    # 사용자가 명시적으로 뒤집음: "질문은 명시적 의도 표시니까 중복억제보다
    # 우선순위가 높아야 한다"). 그래서 질문이면 세션중복 여부와 무관하게
    # 무조건 tier2로 가서 매번 새로 생성된 답을 받는다 -- 같은 질문을 1분
    # 뒤에 또 물어봐도 "조용히 저장됐어요" 같은 정형 문구 대신 실제 답변이
    # 나간다.
    if assessment and assessment["is_question"]:
        return "tier2"

    # "바로 물어보기"로 들어온 요청은 is_question 판정과 무관하게 무조건 tier2로
    # 간다 (2026-10-17, 사용자 지적으로 발견한 갭 수정). is_question은 LLM이
    # 의미로 판단하므로 물음표 없는 애매한 문장이면 False로 오판할 수 있는데,
    # 그 경우 top_score가 높거나 new_concept이 아니면 store_only로 빠져
    # tier2_deep_agent가 아예 안 돌고 사용자가 명시적으로 물어본 질문에 답이
    # 하나도 안 나오는 상태였음 -- "질문엔 무조건 답한다"는 기존 원칙과 같은
    # 결로 is_ask도 여기서 강제.
    if state.get("is_ask"):
        return "tier2"

    # 세션 내에서 방금 다룬 내용과 거의 같으면(그리고 "질문"이 아니면), 중복
    # 힌트를 만들지 않는다 (CLAUDE.md 6번 섹션 "용도1" -- 이미 방금 도와줬으므로).
    # 단, 그 "방금"이 SESSION_DUP_REALLOW_SECONDS보다 오래 전이면 억제를 풀어준다
    # -- 한참 지나 똑같은 내용을 또 썼다는 건 "중복"이 아니라 "아직도 이해가
    # 안 돼서 다시 물어보는 것"일 수 있어서(2026-10-05). is_question 체크를
    # 이 앞으로 옮기면서(위 참고) "질문이든 뭐든"이라는 전제는 더 이상
    # 정확하지 않음 -- 이젠 "질문이 아닌 한" 이 억제가 적용됨.
    session_dup = state.get("session_duplicate_score")
    session_dup_elapsed = state.get("session_duplicate_elapsed_seconds")
    if (
        session_dup is not None
        and session_dup >= SESSION_DUPLICATE_THRESHOLD
        and (session_dup_elapsed is None or session_dup_elapsed <= SESSION_DUP_REALLOW_SECONDS)
    ):
        return "store_only"

    # 힌트 피드백 반영 Phase 2(✓ 억제, 2026-10-08): 가장 비슷한 과거 필기가
    # "이미 알고 있음(✓)" 피드백을 받은 적 있고 그게 FEEDBACK_SUPPRESS_DAYS
    # 이내면, is_new_concept/top_score 같은 나머지 신호와 무관하게 조용히
    # 넘어간다. is_question 체크보다 뒤에 둔 이유: 세션중복과 달리 "며칠 전에
    # 이해했었다"는 사실이 "지금 명시적으로 물어본 것"보다 우선할 이유는
    # 없음 -- 지금 질문했다면 그 질문에는 답해야 함.
    feedback_value = state.get("top_match_feedback_value")
    feedback_days_ago = state.get("top_match_feedback_days_ago")
    if (
        feedback_value == "✓"
        and feedback_days_ago is not None
        and feedback_days_ago <= FEEDBACK_SUPPRESS_DAYS
    ):
        return "store_only"

    # is_new_concept은 계산만 되고 라우팅엔 한 번도 안 쓰이고 있었음(2026-10-02,
    # 6/8일차 리뷰 중 발견 -- CLAUDE.md "라우팅 기준 보강" 변경 이력 참고).
    # LLM이 명시적으로 "새 개념"이라고 판단했다면 top_score/is_abbreviated와
    # 무관하게 보강하는 게 맞다고 판단해 여기 추가. session_dup 체크는 이
    # 조건보다 먼저 와서(위) -- 세션 내에서 방금 다룬 내용이면 "새 개념"
    # 판단이어도 중복 힌트를 또 만들지 않음(일관성 유지).
    #
    # 2026-10-18 수정: top_score 게이트 추가(바로 아래 is_abbreviated 분기와
    # 같은 패턴). 6차 사이클(gpt-6-luna) 분석에서 발견된 구조적 사각지대 --
    # 이 분기엔 top_score 조건이 전혀 없어서, top_score=1.0(사실상 완전
    # 중복)인 반복 템플릿 문구("# daily stand-up" 등)에도 LLM이 is_new_concept
    # =True로 오판하면 그대로 tier2가 발동했다(2026-10-17 CLAUDE.md "6차 사이클
    # 완료" 항목, "구조적 사각지대" 참고 -- 그땐 당장 안 고치고 보완 포인트로만
    # 남겨뒀던 것). `top_score < HIGH_THRESHOLD`를 추가해서, "거의 똑같은
    # 내용"이면 LLM이 새 개념이라고 오판해도 무시하고 store_only 쪽 로직으로
    # 넘어가게 한다 -- 진짜 새로운 내용(top_score가 낮거나 중간)에는 기존과
    # 똑같이 작동한다.
    if (
        assessment
        and assessment["is_new_concept"]
        and (top_score is None or top_score < HIGH_THRESHOLD)
    ):
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

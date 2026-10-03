"""기간(지금은 "날") 단위 요약 생성 -- 구분선의 "지난 시간 복습" 버튼이 호출한다.

2026-10-02 사용자 요청: 필기 작성 단위(우리 경우 날짜)가 끝나는 지점에서
그 단위를 요약해서 보여주고, 그 요약 자체를 저장해 나중에 계층적 요약(day
-> week -> course)의 원재료로 재사용할 수 있게 한다 (CLAUDE.md MVP 범위엔
"계층적 요약"이 애드온으로 빠져있지만, 그 원재료가 될 day 단위 요약만은
지금 쌓아두는 것 -- period_summaries 테이블 참고).

매번 새로 생성하지 않고 캐시한다 -- 같은 날짜를 다시 요청하면 LLM 호출 없이
저장된 결과를 그대로 반환.
"""

from src.db import get_period_summary, get_sim_day_notes, save_period_summary
from src.llm import invoke_chat

_SUMMARY_SYSTEM_PROMPT = (
    "너는 학생의 하루치 수업 필기를 복습용으로 요약해주는 보조 교사야. "
    "주어진 필기 원문(과 곁들여진 에이전트 힌트)을 보고, 그날 다룬 핵심 주제와 "
    "개념을 3~5문장으로 간결하게 정리해줘. 나열식 키워드 말고, 자연스러운 "
    "문장으로. 필기에 없는 내용을 추측해서 채우지 마."
)


def generate_day_summary(course_id: str | None, period_key: str) -> dict:
    """해당 날짜(period_key, YYYYMMDD)의 필기를 요약. 캐시 있으면 재사용.

    반환값: {summary, note_count, tier2_count, cached}
    """
    cached = get_period_summary(course_id, "day", period_key)
    if cached is not None:
        return {
            "summary": cached["summary"],
            "note_count": cached["note_count"],
            "tier2_count": cached["tier2_count"],
            "cached": True,
        }

    notes = get_sim_day_notes(course_id, period_key)
    if not notes:
        return {"summary": "이 날짜에 해당하는 필기를 찾을 수 없어요.", "note_count": 0, "tier2_count": 0, "cached": False}

    lines = "\n".join(f"- {n['content']}" for n in notes)
    tier2_count = sum(1 for n in notes if n.get("tier2_result"))

    response = invoke_chat(
        [
            {"role": "system", "content": _SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": f"{period_key} 필기 원문:\n{lines}"},
        ],
        purpose="day_summary",
    )
    summary = response.content

    save_period_summary(course_id, "day", period_key, summary, len(notes), tier2_count)
    return {"summary": summary, "note_count": len(notes), "tier2_count": tier2_count, "cached": False}

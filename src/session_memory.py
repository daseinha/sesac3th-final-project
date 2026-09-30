"""세션 내 작업기억(InMemory) -- CLAUDE.md 6번 섹션, 용도1(중복 힌트 방지) 구현.

메인 DB(note_chunks)와 역할이 다르다: "이미 정식으로 쌓인 지식"은 DB가,
"오늘 세션에서 방금 다룬 것"은 여기가 담당하고, 세션 종료 시 폐기된다.

세션 경계(언제 시작/끝나는지)는 백엔드가 자동으로 판단하지 않는다 --
호출하는 쪽(session_id)이 넘겨준 값을 그대로 신뢰한다. "언제 세션이
끝나는지"는 제품/UX 설계의 몫이라 (2026-09-30 논의 참고).

프로세스가 계속 살아있는 동안만 유지됨(예: langgraph dev 서버). 일회성
스크립트(uv run python x.py)로 테스트하면 매번 새 프로세스라 세션이 안
이어지므로, 같은 프로세스 안에서 여러 번 호출해서 테스트해야 한다.
"""

from src.db import log_session_event

_session_stores: dict[str, list[list[float]]] = {}


def add_to_session(session_id: str, embedding: list[float]) -> None:
    """이 세션의 작업기억에 필기 하나(임베딩)를 추가."""
    _session_stores.setdefault(session_id, []).append(embedding)
    log_session_event(session_id, "note_added", note_count=len(_session_stores[session_id]))


def check_session_duplicate(session_id: str, embedding: list[float]) -> float | None:
    """이 세션에서 이미 다룬 내용과 얼마나 비슷한지(코사인 유사도 최댓값)를 반환.

    세션이 아직 없거나(첫 필기) 비어있으면 None -- "비교할 게 없다"는 뜻이라
    assess_note의 top_score와 달리 여기선 0.0으로 대체하지 않는다(중복
    여부를 판단하는 신호라, "모른다"와 "안 겹친다"는 다른 의미이므로).
    """
    stored = _session_stores.get(session_id)
    if not stored:
        return None
    return max(_cosine_similarity(embedding, e) for e in stored)


def end_session(session_id: str) -> None:
    """세션 종료 -- 인메모리 저장소 폐기. 없는 session_id를 넘기면 조용히 무시."""
    stored = _session_stores.pop(session_id, None)
    log_session_event(session_id, "session_ended", note_count=len(stored) if stored else 0)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)

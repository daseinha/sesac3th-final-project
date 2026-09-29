"""공개 API 진입점.

CLAUDE.md "백엔드 검증 방식" 섹션에서 정의한 그대로: ingest_note()가 필기
입력의 유일한 진입점이다. 재생 스크립트, (나중에) 프론트엔드 모두 이 함수를
통해서만 필기를 넣는다 -- 임시 테스트 코드가 아니라 실제 서비스 API.
"""

from datetime import datetime
from typing import Literal

from src.app import graph
from src.state import NoteMateState


def ingest_note(
    content: str,
    source: Literal["personal", "base", "stt_reference"],
    medium: Literal["txt", "paper_ocr", "notion", "video_caption"],
    origin_ref: str,
    course_id: str | None = None,
    timestamp: datetime | None = None,
) -> NoteMateState:
    """필기 한 조각을 파이프라인에 투입하고, 처리 완료된 최종 State를 반환한다.

    Args:
        content: 필기 원문.
        source: "personal"(개인 필기) / "base"(개론자료) / "stt_reference"(참고 소스).
        medium: "txt" / "paper_ocr" / "notion" / "video_caption".
        origin_ref: 원본 파일/줄 위치 (원문 보기용 참조).
        course_id: 주차/토픽 단위 식별자 (예: "langgraph_week"). 모르면 None.
        timestamp: 필기 작성 시각. 모르면 None (임의로 채우지 않음).

    Returns:
        그래프 실행이 끝난 최종 State. routing_decision으로 Tier1/2 여부를,
        tier2_result로 실제 힌트 내용을(Tier2일 때만) 확인할 수 있다.
    """
    initial_state = {
        "content": content,
        "source": source,
        "medium": medium,
        "course_id": course_id,
        "note_timestamp": timestamp.isoformat() if timestamp else None,
        "origin_ref": origin_ref,
    }
    return graph.invoke(initial_state)

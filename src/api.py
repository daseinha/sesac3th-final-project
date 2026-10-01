"""공개 API 진입점.

CLAUDE.md "백엔드 검증 방식" 섹션에서 정의한 그대로: ingest_note()가 필기
입력의 유일한 진입점이다. 재생 스크립트, (나중에) 프론트엔드 모두 이 함수를
통해서만 필기를 넣는다 -- 임시 테스트 코드가 아니라 실제 서비스 API.

ingest_file()은 그 위에 한 단계 더 얹은 "업로드 진입점" -- 사용자/프론트엔드
입장에서 txt든 PDF든 같은 함수 하나만 부르면 되고, 확장자 보고 알맞은
어댑터를 자동으로 골라 ingest_note()로 넘기는 역할만 한다.
"""

from datetime import datetime
from pathlib import Path
from typing import Literal

from src.adapters import parse_document_notes, parse_pdf_notes, parse_txt_notes
from src.app import graph
from src.session_memory import end_session  # noqa: F401 -- 재노출 (공개 API로 여기서 접근)
from src.state import NoteMateState


def ingest_note(
    content: str,
    source: Literal["personal", "base", "stt_reference"],
    medium: Literal["txt", "paper_ocr", "notion", "video_caption"],
    origin_ref: str,
    course_id: str | None = None,
    timestamp: datetime | None = None,
    session_id: str | None = None,
) -> NoteMateState:
    """필기 한 조각을 파이프라인에 투입하고, 처리 완료된 최종 State를 반환한다.

    Args:
        content: 필기 원문.
        source: "personal"(개인 필기) / "base"(개론자료) / "stt_reference"(참고 소스).
        medium: "txt" / "paper_ocr" / "notion" / "video_caption".
        origin_ref: 원본 파일/줄 위치 (원문 보기용 참조).
        course_id: 주차/토픽 단위 식별자 (예: "langgraph_week"). 모르면 None.
        timestamp: 필기 작성 시각. 모르면 None (임의로 채우지 않음).
        session_id: 세션 내 작업기억(중복 힌트 방지)에 쓸 식별자. 세션 경계는
            호출하는 쪽이 정한다 -- 안 넘기면 세션 기능 없이 동작(기존과 동일).
            세션이 끝나면 end_session(session_id)로 정리해줄 것.

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
        "session_id": session_id,
    }
    return graph.invoke(initial_state)


# 확장자 -> (어댑터 함수, medium 값). 새 형식을 지원하려면 여기에 한 줄만 추가하면 됨.
# .html/.md는 노션 내보내기라고 가정([잠정] -- 실제 노션 내보내기 파일 보면 재확인 필요).
# .pdf는 parse_pdf_notes()가 텍스트 레이어 유무를 자동 판별해 디지털/이미지형을
# 알아서 분기한다(2026-10-01, GoFullPage 같은 스크롤 캡처 PDF 대응).
_FILE_ADAPTERS = {
    ".txt": (parse_txt_notes, "txt"),
    ".pdf": (parse_pdf_notes, "pdf"),
    ".html": (parse_document_notes, "notion"),
    ".md": (parse_document_notes, "notion"),
}


def ingest_file(
    path: str,
    source: Literal["personal", "base", "stt_reference"],
    course_id: str | None = None,
    session_id: str | None = None,
) -> list[NoteMateState]:
    """업로드된 파일 하나를 확장자로 판별해 알맞은 어댑터로 파싱하고,
    각 조각을 순서대로 ingest_note()에 투입한다.

    사용자/프론트엔드는 "업로드 버튼 하나"로 이 함수만 호출하면 되고,
    txt/PDF 등 형식 차이는 이 함수 내부(어댑터 선택)에서만 처리된다.
    """
    ext = Path(path).suffix.lower()
    if ext not in _FILE_ADAPTERS:
        supported = ", ".join(_FILE_ADAPTERS)
        raise ValueError(f"지원하지 않는 파일 형식: {ext} (지원: {supported})")

    parse_fn, medium = _FILE_ADAPTERS[ext]
    notes = parse_fn(path)

    return [
        ingest_note(
            content=note.content,
            source=source,
            medium=medium,
            origin_ref=note.origin_ref,
            course_id=course_id,
            timestamp=note.timestamp,
            session_id=session_id,
        )
        for note in notes
    ]

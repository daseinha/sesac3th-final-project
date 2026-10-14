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
    client_line_id: str | None = None,
    dry_run: bool = False,
    inserted_out_of_place: bool = False,
    allow_general_knowledge: bool = True,
    is_ask: bool = False,
    upload_label: str | None = None,
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
        client_line_id: 프론트엔드가 매긴 "필기 줄" 식별자. 같은 줄을 고쳐서
            여러 번 제출해도 같은 값을 넘기면, get_all_notes()가 복원할 때
            최신 버전 하나만 보여준다(안 넘기면 매번 별개의 줄로 취급됨 --
            재생 스크립트/업로드처럼 한 번만 제출되는 호출은 그냥 생략하면 됨).
        dry_run: True면 검색/판단/힌트 생성은 기존 데이터를 근거로 실제로
            돌아가지만, 그 결과를 DB에 저장하지도 세션 작업기억에 등록하지도
            않는다(2026-10-10, 테스트 입력이 RAG 코퍼스를 오염시키지 않으면서도
            힌트 생성 품질을 확인할 수 있게).
        inserted_out_of_place: 사용자가 타임라인 중 과거 세션/날짜 블록으로
            스크롤해서 그 자리에 입력한 경우 True. 판단/라우팅엔 안 쓰이고
            그대로 저장만 됨 -- 화면 표시용([[deferred-historical-note-insertion]]).
        allow_general_knowledge: False면 Tier2가 참고할 근거(강의자료/내 필기)를
            하나도 못 찾았을 때 LLM 자기 지식만으로 힌트를 만드는 걸 건너뛴다
            (2026-10-12, 힌트 남발 완화 + 근거 유무별 비교테스트용). 기본값
            True는 기존 동작(일반 지식으로라도 답변) 그대로.
        is_ask: "바로 물어보기"로 들어온 요청이면 True(서버가 origin_prefix로
            자동 판별해서 넘김). 근거 부족 + True일 때만 웹검색(Tavily)이
            개입한다(2026-10-12) -- 조용한 힌트 경로에는 절대 안 씀.
        upload_label: 자료 업로드 UI에서 사용자가 고르거나 새로 만든 분류명
            (2026-10-12). 판단/라우팅엔 안 쓰이고 그대로 저장만 됨.

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
        "client_line_id": client_line_id,
        "dry_run": dry_run,
        "inserted_out_of_place": inserted_out_of_place,
        "allow_general_knowledge": allow_general_knowledge,
        "is_ask": is_ask,
        "upload_label": upload_label,
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
    upload_label: str | None = None,
) -> list[NoteMateState]:
    """업로드된 파일 하나를 확장자로 판별해 알맞은 어댑터로 파싱하고,
    각 조각을 순서대로 ingest_note()에 투입한다.

    사용자/프론트엔드는 "업로드 버튼 하나"로 이 함수만 호출하면 되고,
    txt/PDF 등 형식 차이는 이 함수 내부(어댑터 선택)에서만 처리된다.

    upload_label: 업로드 UI에서 사용자가 고르거나 새로 만든 분류명(2026-10-12).
        이 파일에서 나온 모든 조각에 똑같이 붙는다.
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
            upload_label=upload_label,
        )
        for note in notes
    ]

"""종이 필기 스캔본(61페이지) 재인제스트 -- 2026-10-14.

9/30에 한 번 성공적으로 인제스트했던 데이터(token_usage 테이블 기록으로 확인됨,
paper_ocr_page=59건/paper_ocr_reference_embedding=96건/paper_ocr_manual_claude=3건)
가 10/1 중복 적재 사고 복구 때 같은 폴더(data/personal/)를 겨냥한 대량 DELETE에
휩쓸려 사라진 것으로 추정 -- 원본 파일과 파싱 로직은 그대로 있어서 재인제스트.

표준 ingest_file() 대신 parse_vision_pdf_notes()를 직접 호출하고 ingest_note()에
medium="paper_ocr"을 명시한다 -- ingest_file()의 _FILE_ADAPTERS[".pdf"]는 항상
medium="pdf"로 뭉뚱그려서 디지털 문서와 손글씨 스캔을 구분 못 하기 때문
(CLAUDE.md 2026-10-14 저녁 추가 참고).

실패한 페이지(비전 LLM이 다이어그램 폭주 등으로 실패)는 parse_vision_pdf_notes()
자체가 이미 failed_ingestions 테이블에 기록해준다 -- 이 스크립트 실행 후
`SELECT * FROM failed_ingestions`로 확인해서 수동 보완(Claude 직접 전사)하면 됨.
"""

from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

from src.adapters import parse_vision_pdf_notes  # noqa: E402
from src.api import ingest_note  # noqa: E402

_PATH = "data/personal/Scan_20260930_145133.pdf"
_COURSE_ID = "sesac_bootcamp"


def main() -> None:
    notes = parse_vision_pdf_notes(_PATH)
    print(f"OCR 완료: {len(notes)}개 페이지에서 텍스트 추출됨. 인제스트 시작...")

    log_lines = []
    for i, note in enumerate(notes, start=1):
        state = ingest_note(
            content=note.content,
            source="personal",
            medium="paper_ocr",
            origin_ref=note.origin_ref,
            course_id=_COURSE_ID,
            timestamp=note.timestamp,
        )
        log_lines.append(
            f"[{i}/{len(notes)}] {note.origin_ref} "
            f"timestamp={note.timestamp} routing={state.get('routing_decision')}"
        )
        print(log_lines[-1])

    with open("reingest_paper_notes_result.log", "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines))
    print(f"완료: {len(notes)}건 인제스트. 로그: reingest_paper_notes_result.log")


if __name__ == "__main__":
    main()

"""필기 소스별 어댑터 -- 서로 다른 원본 형식을 ingest_note()가 받는 표준 입력으로 변환.

지금은 txt 어댑터만 있음 (CLAUDE.md 개발 순서: "txt부터, OCR/노션은 이후").
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass
class NoteInput:
    """ingest_note() 호출에 필요한 필드만 담은 어댑터 공통 출력."""

    content: str
    timestamp: datetime | None
    origin_ref: str


def parse_txt_notes(path: str) -> list[NoteInput]:
    """빈 줄로 구분된 txt 필기 파일을 파싱한다.

    파일 형식 (한 문단 = 필기 한 조각):
        [YYYY-MM-DD HH:MM]   <- 선택. 없으면 timestamp=None
        필기 내용 (여러 줄 가능)

        (빈 줄로 다음 조각과 구분)

    실제 사용자 필기 파일의 정확한 형식은 아직 안 봐서 [잠정]으로 정한 규칙이다 --
    실제 파일을 보면 이 파서를 맞춰서 고쳐야 할 수 있음.
    """
    text = open(path, encoding="utf-8").read()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    notes: list[NoteInput] = []
    for i, paragraph in enumerate(paragraphs):
        lines = paragraph.splitlines()
        timestamp = None
        content_lines = lines

        first_line = lines[0].strip()
        if first_line.startswith("[") and first_line.endswith("]"):
            try:
                timestamp = datetime.strptime(first_line[1:-1], "%Y-%m-%d %H:%M")
                content_lines = lines[1:]
            except ValueError:
                pass  # 타임스탬프 형식이 아니면 그냥 본문의 일부로 취급

        content = "\n".join(content_lines).strip()
        if not content:
            continue

        notes.append(
            NoteInput(
                content=content,
                timestamp=timestamp,
                origin_ref=f"{path}:{i}",
            )
        )
    return notes

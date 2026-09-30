"""필기 소스별 어댑터 -- 서로 다른 원본 형식을 ingest_note()가 받는 표준 입력으로 변환.

txt(개인 필기) + Docling 기반 문서 어댑터(PDF/노션 내보내기 등 수업자료) 둘 다 있음.
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


def parse_document_notes(path: str) -> list[NoteInput]:
    """Docling으로 문서(PDF, 노션 내보내기 HTML/MD 등)를 파싱해 청크로 변환.

    HybridChunker.contextualize()로 각 청크 앞에 소속 헤딩 경로를 붙인다
    (예: "1주차: LangGraph 기초 > 조건부 엣지\\n<본문>") -- 헤딩 문맥이 있어야
    임베딩이 "이게 무슨 얘기인지" 더 잘 담아서, 검색 품질이 좋아진다.

    문서 자체엔 작성 시각이 없으므로 timestamp는 항상 None -- 모르는 걸 임의로
    채우지 않는다는 CLAUDE.md 원칙 그대로.

    docling은 무거운 의존성(torch 등)이라, txt만 쓰는 경로에서 매번 로드되지
    않도록 함수 안에서 지연 import한다.
    """
    from docling.chunking import HybridChunker
    from docling.document_converter import DocumentConverter

    converter = DocumentConverter()
    result = converter.convert(path)
    doc = result.document

    chunker = HybridChunker()
    chunks = list(chunker.chunk(doc))

    return [
        NoteInput(
            content=chunker.contextualize(chunk),
            timestamp=None,
            origin_ref=f"{path}:chunk{i}",
        )
        for i, chunk in enumerate(chunks)
    ]

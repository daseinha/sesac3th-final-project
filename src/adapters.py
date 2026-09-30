"""필기 소스별 어댑터 -- 서로 다른 원본 형식을 ingest_note()가 받는 표준 입력으로 변환.

txt(개인 필기) + Docling 기반 문서 어댑터(PDF/노션 내보내기 등 수업자료) 둘 다 있음.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# 파일명 앞 8자리 숫자를 그날 날짜로 인식 (예: "20260702 (목) 4회차 수업일지.txt")
_FILENAME_DATE_RE = re.compile(r"^(\d{8})")


@dataclass
class NoteInput:
    """ingest_note() 호출에 필요한 필드만 담은 어댑터 공통 출력."""

    content: str
    timestamp: datetime | None
    origin_ref: str


def parse_txt_notes(path: str) -> list[NoteInput]:
    """하루 단위 필기 일지(txt)를 파싱한다.

    실제 사용자 파일 형식(2026-09-30 확인): 파일 하나 = 하루치 필기 전체.
    파일 안에는 타임스탬프가 따로 없고, 파일명 앞 8자리 숫자(YYYYMMDD)가 그날
    날짜다. 그 하루 동안 쓴 필기가 쭉 이어지며, 빈 줄로 대략적인 단락(주제)이
    나뉜다.

    ([잠정] 이전 버전은 "조각마다 [YYYY-MM-DD HH:MM] 인라인 표시"를 가정했는데,
    실제 파일과 달라서 이 버전으로 교체함 -- CLAUDE.md 변경 이력 참고. 파일명에
    날짜가 없으면 timestamp=None으로 남긴다.)
    """
    match = _FILENAME_DATE_RE.match(Path(path).stem)
    file_date = datetime.strptime(match.group(1), "%Y%m%d") if match else None

    text = open(path, encoding="utf-8").read()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    return [
        NoteInput(content=paragraph, timestamp=file_date, origin_ref=f"{path}:{i}")
        for i, paragraph in enumerate(paragraphs)
    ]


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

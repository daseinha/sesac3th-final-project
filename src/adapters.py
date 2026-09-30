"""필기 소스별 어댑터 -- 서로 다른 원본 형식을 ingest_note()가 받는 표준 입력으로 변환.

txt(개인 필기) + Docling 기반 문서 어댑터(PDF/노션 내보내기 등 수업자료) 둘 다 있음.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# 파일명 앞 8자리 숫자를 그날 날짜로 인식 (예: "20260702 (목) 4회차 수업일지.txt")
_FILENAME_DATE_RE = re.compile(r"^(\d{8})")

# 한글/영문/숫자가 이 정도는 있어야 "내용 있는 청크"로 취급. 구분선(------)만
# 있거나 텅 빈 문단은 걸러낸다 -- 실제 필기 파일에서 파일마다 반복 확인된 노이즈.
_SUBSTANTIAL_CONTENT_RE = re.compile(r"[가-힣a-zA-Z0-9]{3,}")


def _has_substantial_content(text: str) -> bool:
    return bool(_SUBSTANTIAL_CONTENT_RE.search(text))


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
    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip() and _has_substantial_content(p)
    ]

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

    OCR/표구조 인식은 기본 꺼둔다 -- 우리가 다루는 문서는 노션 등에서 복붙한
    단순 텍스트(표/스캔 이미지 없음)라 이 두 단계가 필요 없고, 꺼두면 속도가
    ~4배 빨라짐(4페이지 PDF 기준 21.2초 -> 5.5초, 2026-09-30 실측). 표/스캔
    이미지가 있는 문서를 다뤄야 하면 그때 이 옵션을 켜거나(속도 손해 감수) LlamaParse로
    전환 -- CLAUDE.md 기술 스펙의 "Docling이 구조를 못 잡는 문서만 LlamaParse" 기준 그대로.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.chunking import HybridChunker
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = False

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )
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

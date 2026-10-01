"""필기 소스별 어댑터 -- 서로 다른 원본 형식을 ingest_note()가 받는 표준 입력으로 변환.

txt(개인 필기) + Docling 기반 문서 어댑터(PDF/노션 내보내기 등 수업자료) 둘 다 있음.
"""

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# 파일명 앞 8자리 숫자를 그날 날짜로 인식 (예: "20260702 (목) 4회차 수업일지.txt")
_FILENAME_DATE_RE = re.compile(r"^(\d{8})")

# 파일 "내용" 안에서 날짜 구분 헤더를 인식 (예: "20260715 수 (서초 새싹 13일차) 점심 (...)").
# 하루 단위 파일이든, 여러 날짜가 합쳐진 합본 파일이든 이 패턴이 나올 때마다
# "현재 날짜"를 갱신하는 방식으로 둘 다 같은 로직으로 처리한다.
_DAY_HEADER_RE = re.compile(r"^(\d{8})\s*[월화수목금토일]", re.MULTILINE)

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
    """필기 일지(txt)를 파싱한다. 파일 하나 = 하루든, 여러 날짜가 합쳐진
    합본이든 둘 다 처리한다.

    실제 사용자 파일 형식(2026-09-30 확인): 타임스탬프가 조각마다 따로 없고,
    "20260715 수 (...) 점심 (...)" 같은 날짜 헤더가 하루 분량 시작마다 나온다.
    빈 줄로 대략적인 단락(주제)이 나뉜다.

    처리 방식: 빈 줄로 전체를 문단 나눈 뒤, 순서대로 훑으면서 날짜 헤더 패턴을
    만나면 "현재 날짜"를 그 값으로 갱신 -- 이후 문단들은 다음 헤더가 나올 때까지
    그 날짜를 그대로 물려받는다. 파일명에 날짜가 있으면(기존 "파일 하나=하루"
    파일들) 그걸 시작 기본값으로 쓰고, 파일 맨 앞에 헤더가 없어도 자연스럽게
    맞아떨어진다. 합본 파일처럼 파일명에 날짜가 없으면 첫 헤더를 만나기 전까지는
    timestamp=None.

    ([잠정] 이전 버전은 "조각마다 [YYYY-MM-DD HH:MM] 인라인 표시"를 가정했는데,
    실제 파일과 달라서 이 버전으로 교체함 -- CLAUDE.md 변경 이력 참고.)
    """
    match = _FILENAME_DATE_RE.match(Path(path).stem)
    current_date = datetime.strptime(match.group(1), "%Y%m%d") if match else None

    text = open(path, encoding="utf-8").read()
    paragraphs = [
        p.strip()
        for p in text.split("\n\n")
        if p.strip() and _has_substantial_content(p)
    ]

    notes: list[NoteInput] = []
    for i, paragraph in enumerate(paragraphs):
        header_match = _DAY_HEADER_RE.search(paragraph)
        if header_match:
            current_date = datetime.strptime(header_match.group(1), "%Y%m%d")
            # 헤더 문단(날짜/장소/점심메뉴) 자체는 학습 내용이 아니라 메타정보라
            # 청크로 남기지 않는다 -- 날짜 갱신에만 쓰고 건너뜀. 2026-09-30 실측:
            # 이런 헤더 조각이 경계 구간(top_score 0.7~0.9)에서 Tier2를 낭비시킨
            # 사례의 대부분(62%, 45개 중 28개)이었음. CLAUDE.md 변경 이력 참고.
            continue
        notes.append(NoteInput(content=paragraph, timestamp=current_date, origin_ref=f"{path}:{i}"))
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


_VISION_OCR_PROMPT = (
    "이 이미지는 문서 페이지입니다(손글씨 필기 스캔본이거나, 웹페이지를 캡처한 "
    "자료일 수 있습니다). 보이는 텍스트를 최대한 정확하게 옮겨써주세요 -- 제목/"
    "목록/코드블록 같은 구조가 있으면 마크다운으로 살려서 옮기세요. 읽을 수 없는 "
    "부분은 [판독불가]로 표시하세요. 본문에 날짜(예: 6/29, 2026.07.02, 20260831 "
    "등)가 명확히 보이면 detected_date에 YYYY-MM-DD 형식으로 채우고, 안 보이거나 "
    "확실하지 않으면(예: 강의자료처럼 날짜 자체가 없는 문서) null로 두세요."
)


def parse_vision_pdf_notes(path: str) -> list[NoteInput]:
    """텍스트 레이어가 없는 PDF를 페이지 단위로 비전 LLM(gpt-4.1-mini)으로 읽어서 변환.

    원래 손글씨 스캔본용으로 만들었는데(전통 OCR 인식률이 낮고 날짜도 거의 못
    읽음, 61페이지 중 3개만 성공), **GoFullPage 같은 스크롤 캡처 도구로 만든
    "이미지형" PDF에도 그대로 재사용**한다 -- 2026-10-01 실측: 이런 PDF는
    텍스트 레이어가 없는 건 물론, Docling의 OCR(RapidOCR/EasyOCR 둘 다)마저
    완전히 실패함(다크모드 코드블록 등에서 레이아웃 분석 자체가 텍스트를 하나도
    못 찾음, 4페이지 전부 빈 결과). 비전 LLM은 거부 없이 마크다운 구조까지
    살려서 정확하게 옮김.

    실측 비교(2026-09-30, CLAUDE.md 참고)에서 `gpt-4o`는 API키/인증 관련
    페이지를 60%(5개 중 3개) 거부했지만 `gpt-4.1-mini`는 거부 없이 처리함.

    페이지마다 날짜 인식을 구조화 출력으로 같이 시도한다 -- 못 읽으면 None
    (임의로 채우지 않음, 강의자료처럼 애초에 날짜가 없는 문서도 자연히 None).
    내용 없는 페이지는 스킵.
    """
    import base64
    import io

    import pypdfium2 as pdfium
    from pydantic import BaseModel, Field

    from src.llm import invoke_structured

    class _PageOcrResult(BaseModel):
        transcription: str = Field(description="이미지 속 손글씨 텍스트를 최대한 정확하게 옮겨쓴 것")
        detected_date: str | None = Field(
            default=None,
            description="본문에서 명확히 식별되는 날짜가 있으면 YYYY-MM-DD 형식으로, 없거나 불확실하면 null",
        )

    pdf = pdfium.PdfDocument(path)
    notes: list[NoteInput] = []

    for i, page in enumerate(pdf, start=1):
        bitmap = page.render(scale=2.0)
        buf = io.BytesIO()
        bitmap.to_pil().save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        try:
            result: _PageOcrResult = invoke_structured(
                _PageOcrResult,
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _VISION_OCR_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
                            },
                        ],
                    }
                ],
                purpose="vision_pdf_page",
                max_tokens=2000,
            )
        except Exception as e:  # noqa: BLE001 -- 페이지 하나 실패해도 나머지는 계속 처리
            print(f"[page {i}] 실패, 스킵: {type(e).__name__}: {e}")
            continue

        if not _has_substantial_content(result.transcription):
            continue

        timestamp = None
        if result.detected_date:
            try:
                timestamp = datetime.strptime(result.detected_date, "%Y-%m-%d")
            except ValueError:
                pass  # 형식이 안 맞으면 그냥 None으로 둠 -- 임의로 채우지 않음

        notes.append(
            NoteInput(content=result.transcription, timestamp=timestamp, origin_ref=f"{path}:page{i}")
        )

    return notes


def parse_pdf_notes(path: str) -> list[NoteInput]:
    """PDF 파싱 진입점 -- 텍스트 레이어 유무를 자동 판별해 알맞은 방식으로 처리.

    같은 `.pdf` 확장자 안에 "디지털 문서"(Notion 복붙, 인쇄->PDF 등 실제 텍스트
    있음)와 "이미지형"(손글씨 스캔, GoFullPage 같은 스크롤 캡처 -- 텍스트 레이어
    없음) 둘 다 있어서 폴더/이름 규칙으로 미리 구분하는 대신, **일단 빠른
    방법(Docling, OCR 끔)을 시도하고 결과가 비어있으면(=텍스트 레이어가 없다는
    뜻) 비전 LLM으로 자동 전환**하는 방식으로 판별한다 (2026-10-01).
    """
    notes = parse_document_notes(path)
    if notes:
        return notes
    return parse_vision_pdf_notes(path)

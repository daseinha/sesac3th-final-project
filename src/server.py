"""프론트엔드(notemate_mockup.html)가 호출할 얇은 FastAPI 래퍼.

`langgraph dev`(그래프 디버깅/트레이스용, src/app.py가 띄우는 것과 별개)와는
무관하다 -- 이 서버는 기존에 이미 검증된 `ingest_note()`/`ingest_file()`을
그대로 import해서 직접 호출할 뿐, LangGraph 서버에 HTTP로 요청을 보내지
않는다(2026-10-02, CLAUDE.md "프론트엔드 아키텍처 재검토" 참고).

이 서버 하나가 API 엔드포인트 + 정적 목업 페이지(`notemate_mockup.html`)
서빙까지 같이 담당한다 -- 같은 출처(origin)라 CORS 설정이 필요 없다.
"""

import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# uvicorn --reload는 서브프로세스로 뜨는데, 그 프로세스의 작업 디렉터리 기준
# 자동 탐색이 안 먹힐 때가 있어서(2026-10-02 실측) 절대경로로 명시한다.
load_dotenv(dotenv_path=str(Path(__file__).resolve().parent.parent / ".env"))

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse

from src.api import ingest_file, ingest_note
from src.db import get_today_hints, get_today_notes
from src.summarize import generate_day_summary

_DEFAULT_COURSE_ID = "sesac_bootcamp"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_MOCKUP_PATH = _PROJECT_ROOT / "notemate_mockup.html"
# 힌트 피드백 기록 (2026-10-07 결정: 1단계는 로그 파일로만 쌓고, 라우팅에는 연결하지 않음).
# 사이클 테스트 후 DB 테이블로 옮기고 라우터에 연결해서 평가한다.
_HINT_FEEDBACK_LOG = _PROJECT_ROOT / "data" / "hint_feedback_log.jsonl"
_FEEDBACK_VALUES = {"+", "−", "-", "✓"}

app = FastAPI(title="NoteMate API")


def _append_hint_event(record: dict) -> None:
    _HINT_FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(_HINT_FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@app.get("/")
def serve_mockup() -> FileResponse:
    """목업 페이지를 그대로 서빙 -- Streamlit/iframe 없이 이 서버가 직접 호스팅."""
    return FileResponse(_MOCKUP_PATH)


@app.post("/api/ingest_note")
def api_ingest_note(
    content: str = Form(...),
    course_id: str = Form(_DEFAULT_COURSE_ID),
    session_id: str = Form(...),
    origin_prefix: str = Form("live"),
    client_line_id: str | None = Form(None),
) -> dict:
    """필기 한 조각(문단/줄 단위)을 받아 파이프라인에 투입.

    타임스탬프는 호출 시점(지금)으로 고정 -- 재생 스크립트와 달리 실시간
    입력이라 "모르면 None" 원칙이 적용 안 됨(CLAUDE.md의 timestamp 원칙은
    backfill/업로드 자료처럼 작성 시각을 알 수 없는 경우를 위한 것).

    `origin_prefix`: "live"(필기창에서 직접 입력, 기본값) vs "ask"("지금 바로
    물어보기"로 들어온 즉석 질문). 둘 다 source=personal로 똑같이 저장은
    되지만(통계/RAG에는 둘 다 필요), `get_today_notes()`가 이 접두사로
    "ask"는 걸러내서 필기창 복원 대상에서 뺀다 -- 안 그러면 새로고침할 때
    질문이 마치 필기 한 줄처럼 복원돼버림(2026-10-02 실측 버그).

    `client_line_id`: 프론트엔드가 필기 줄 하나에 매긴 식별자. 같은 줄을
    고쳐서 여러 번 제출해도 같은 값이 넘어오면, get_today_notes()가 그
    줄의 최신 버전만 복원한다(안 넘기면 매번 새 줄로 취급돼서, 줄을 고쳐
    쓸수록 새로고침할 때 과거 버전까지 전부 중복으로 쌓여 보이던 버그가
    있었다 -- 2026-10-02 실측).
    """
    state = ingest_note(
        content=content,
        source="personal",
        medium="txt",
        origin_ref=f"{origin_prefix}:{session_id}:{datetime.now().isoformat()}",
        course_id=course_id,
        timestamp=datetime.now(),
        session_id=session_id,
        client_line_id=client_line_id,
    )
    if state.get("tier2_result"):
        # 노출(shown) 기록: 힌트가 실제로 만들어져 화면에 전달된 순간. 분모 용도.
        _append_hint_event({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "event_type": "shown",
            "course_id": course_id,
            "session_id": session_id,
            "origin_prefix": origin_prefix,
            "client_line_id": client_line_id,
            "hint_text": state.get("tier2_result"),
            "top_score": state.get("top_score"),
            "routing_decision": state.get("routing_decision"),
        })
    return {
        "routing_decision": state.get("routing_decision"),
        "top_score": state.get("top_score"),
        "is_question": state.get("is_question"),
        "tier2_result": state.get("tier2_result"),
    }


@app.post("/api/hint_feedback")
def api_hint_feedback(
    course_id: str = Form(_DEFAULT_COURSE_ID),
    session_id: str = Form(...),
    hint_text: str = Form(...),
    value: str = Form(...),
    kind: str = Form("hint"),
) -> dict:
    """힌트(hint) 또는 바로 물어보기 답변(ask)의 버튼(+ / − / ✓) 클릭을 기록한다. 라우팅에는 연결하지 않는다."""
    if value not in _FEEDBACK_VALUES:
        return {"ok": False, "error": "unknown value"}
    _append_hint_event({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "event_type": "feedback",
        "kind": kind if kind in ("hint", "ask") else "hint",
        "course_id": course_id,
        "session_id": session_id,
        "value": value,
        "hint_text": hint_text,
    })
    return {"ok": True}


@app.post("/api/ingest_file")
async def api_ingest_file(
    file: UploadFile = File(...),
    source: str = Form("base"),
    course_id: str = Form(_DEFAULT_COURSE_ID),
) -> dict:
    """업로드 버튼이 호출 -- 받은 파일을 data/<source>/에 저장한 뒤 ingest_file()로 처리.

    data/<source>/ 아래에 실물 파일로 남겨두는 이유: 기존 `ingest_folder.py`
    폴더 컨벤션과 그대로 호환되고(나중에 일괄 재처리/검증 스크립트가 그대로
    먹힘), 원본을 디스크에 보존해 문제 생기면 재인제스트 가능.
    """
    dest_dir = _PROJECT_ROOT / "data" / source
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / file.filename

    with open(dest_path, "wb") as f:
        f.write(await file.read())

    results = ingest_file(str(dest_path), source=source, course_id=course_id)
    return {"chunks_saved": len(results), "filename": file.filename}


@app.get("/api/today_hints")
def api_today_hints(course_id: str = _DEFAULT_COURSE_ID) -> list[dict]:
    """오른쪽 "오늘의 요약" 패널용 -- 오늘 Tier2가 개입한 힌트 목록."""
    hints = get_today_hints(course_id)
    return [
        {
            "content": h["content"],
            "tier2_result": h["tier2_result"],
            "created_at": h["created_at"].isoformat(),
        }
        for h in hints
    ]


@app.get("/api/today_notes")
def api_today_notes(course_id: str = _DEFAULT_COURSE_ID) -> list[dict]:
    """페이지를 새로고침해도 오늘 실제로 쓴 필기가 그대로 복원되도록 -- 가짜
    데모 문단 대신 오늘 입력된 personal 필기 전체(Tier1/2 가리지 않고)를 반환."""
    notes = get_today_notes(course_id)
    return [
        {
            "content": n["content"],
            "tier2_result": n["tier2_result"],
            "created_at": n["created_at"].isoformat(),
            "client_line_id": n["client_line_id"],
            "origin_ref": n["origin_ref"],
        }
        for n in notes
    ]


@app.post("/api/day_summary")
def api_day_summary(period_key: str = Form(...), course_id: str = Form(_DEFAULT_COURSE_ID)) -> dict:
    """필기 타임라인의 날짜 구분선 "지난 시간 복습" 버튼이 호출.

    period_key: YYYYMMDD. 같은 날짜를 또 요청하면 재생성 없이 캐시된 요약을
    그대로 반환(src/summarize.py 참고).
    """
    return generate_day_summary(course_id, period_key)

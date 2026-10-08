"""프론트엔드(notemate_mockup.html)가 호출할 얇은 FastAPI 래퍼.

`langgraph dev`(그래프 디버깅/트레이스용, src/app.py가 띄우는 것과 별개)와는
무관하다 -- 이 서버는 기존에 이미 검증된 `ingest_note()`/`ingest_file()`을
그대로 import해서 직접 호출할 뿐, LangGraph 서버에 HTTP로 요청을 보내지
않는다(2026-10-02, CLAUDE.md "프론트엔드 아키텍처 재검토" 참고).

이 서버 하나가 API 엔드포인트 + 정적 목업 페이지(`notemate_mockup.html`)
서빙까지 같이 담당한다 -- 같은 출처(origin)라 CORS 설정이 필요 없다.
"""

import json
import os
import threading
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# uvicorn --reload는 서브프로세스로 뜨는데, 그 프로세스의 작업 디렉터리 기준
# 자동 탐색이 안 먹힐 때가 있어서(2026-10-02 실측) 절대경로로 명시한다.
load_dotenv(dotenv_path=str(Path(__file__).resolve().parent.parent / ".env"))

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import FileResponse

from src.api import ingest_file, ingest_note
from src.db import (
    delete_session_notes,
    get_all_notes,
    get_deleted_sessions,
    get_today_hints,
    insert_deleted_session,
    record_hint_feedback,
)
from src.session_memory import end_session
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


# 개발 중 이 정적 파일들을 수시로 고쳐서 바로 다시 서빙하므로, 브라우저가
# Cache-Control 없이 알아서 판단(heuristic caching)하게 두지 않고 "매번
# 서버에 확인해라"라고 명시한다 -- 그래야 일반 새로고침(F5)만으로도 항상
# 최신 파일이 뜨고, 하드 리프레시(Ctrl+Shift+R)를 매번 따로 안 해도 됨
# (2026-10-08, 사용자 질문: "새로고침 할때마다 하드리셋을 해야하는 거야?").
# no-store가 아니라 no-cache인 이유: no-cache는 매번 서버에 재검증(조건부
# GET)만 요구하고, 안 바뀌었으면 여전히 304로 가볍게 응답 가능함.
_NO_CACHE_HEADERS = {"Cache-Control": "no-cache"}


@app.get("/")
def serve_mockup() -> FileResponse:
    """목업 페이지를 그대로 서빙 -- Streamlit/iframe 없이 이 서버가 직접 호스팅."""
    return FileResponse(_MOCKUP_PATH, headers=_NO_CACHE_HEADERS)


@app.get("/review/run3")
def serve_run3_review() -> FileResponse:
    """3차 재생 테스트(2026-10-07) 전체 필기+힌트를 사람이 훑어보는 읽기 전용 리뷰 페이지.

    같은 서버가 서빙해야 페이지 안 피드백 버튼의 fetch('/api/hint_feedback')가
    같은 출처(same-origin)로 나가서 CORS 문제 없이 동작한다. 서비스 화면
    (notemate_mockup.html)과는 무관한 별도 분석용 파일(data/ 아래, gitignore 대상).
    """
    return FileResponse(
        _PROJECT_ROOT / "data" / "review_run3_all_notes_2026-10-07.html",
        headers=_NO_CACHE_HEADERS,
    )


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
    되지만(통계/RAG에는 둘 다 필요), `get_all_notes()`가 이 접두사로
    "ask"는 걸러내서 필기창 복원 대상에서 뺀다 -- 안 그러면 새로고침할 때
    질문이 마치 필기 한 줄처럼 복원돼버림(2026-10-02 실측 버그).

    `client_line_id`: 프론트엔드가 필기 줄 하나에 매긴 식별자. 같은 줄을
    고쳐서 여러 번 제출해도 같은 값이 넘어오면, get_all_notes()가 그
    줄의 최신 버전만 복원한다(안 넘기면 매번 새 줄로 취급돼서, 줄을 고쳐
    쓸수록 새로고침할 때 과거 버전까지 전부 중복으로 쌓여 보이던 버그가
    있었다 -- 2026-10-02 실측).
    """
    origin_ref = f"{origin_prefix}:{session_id}:{datetime.now().isoformat()}"
    state = ingest_note(
        content=content,
        source="personal",
        medium="txt",
        origin_ref=origin_ref,
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
        # 2026-10-08: 프론트엔드가 피드백 버튼 클릭 시 이 origin_ref를 같이 보내야
        # note_chunks.feedback_value에 기록되고, Phase 2(✓ 억제)가 작동한다.
        "origin_ref": origin_ref,
    }


@app.post("/api/hint_feedback")
def api_hint_feedback(
    course_id: str = Form(_DEFAULT_COURSE_ID),
    session_id: str = Form(...),
    hint_text: str = Form(...),
    value: str = Form(...),
    kind: str = Form("hint"),
    origin_ref: str | None = Form(None),
) -> dict:
    """힌트(hint) 또는 바로 물어보기 답변(ask)의 버튼(+ / − / ✓) 클릭을 기록한다. 라우팅에는 연결하지 않는다.

    `origin_ref`: 이 힌트가 달린 필기 줄의 고유 식별자(note_chunks.origin_ref,
    UNIQUE 제약 있음) -- 2026-10-08 추가. 로그 파일(복원용)뿐 아니라
    `note_chunks.feedback_value`/`feedback_at`에도 기록해서, 힌트 피드백
    반영 Phase 2(✓ 억제)가 실제로 작동하게 한다(`src/routers.py`
    `FEEDBACK_SUPPRESS_DAYS` 참고) -- origin_ref가 없으면(구버전 호출 등)
    로그만 남고 반영은 안 됨.
    """
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
        "origin_ref": origin_ref,
    })
    reflected = record_hint_feedback(origin_ref, value) if origin_ref else False
    return {"ok": True, "reflected": reflected}


@app.get("/api/hint_feedback_state")
def api_hint_feedback_state() -> dict:
    """origin_ref별로 마지막에 선택된 피드백 값만 뽑아서 돌려준다.

    로그 파일을 그대로 쌓아두는 구조라(이벤트 로그, 덮어쓰기 없음) 같은
    origin_ref에 여러 번 클릭했으면 가장 나중 것이 유효한 선택이다 -- 파일을
    위에서부터 순서대로 읽어 덮어써서 자연히 "마지막 값"만 남긴다. 리뷰
    페이지(`/review/run3`)가 새로고침 후에도 선택 표시를 복원하는 데 쓴다.
    """
    state: dict[str, str] = {}
    if _HINT_FEEDBACK_LOG.exists():
        with open(_HINT_FEEDBACK_LOG, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                ref = record.get("origin_ref")
                if record.get("event_type") == "feedback" and ref:
                    state[ref] = record["value"]
    return state


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


@app.get("/api/all_notes")
def api_all_notes(course_id: str = _DEFAULT_COURSE_ID) -> list[dict]:
    """페이지를 새로고침하거나 서비스를 재시작해도 이 과목의 필기 전체가 그대로
    복원되도록(2026-10-08, "종료 후 재시작 시 기존 필기가 떠있어야 한다"는
    요청 반영 -- 구 이름 api_today_notes, 날짜 제한 제거) -- 가짜 데모 문단
    대신 입력된 personal 필기 전체(Tier1/2 가리지 않고)를 반환."""
    notes = get_all_notes(course_id)
    return [
        {
            "content": n["content"],
            "tier2_result": n["tier2_result"],
            "created_at": n["created_at"].isoformat(),
            "client_line_id": n["client_line_id"],
            "origin_ref": n["origin_ref"],
            "session_id": n["session_id"],
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


@app.post("/api/delete_session")
def api_delete_session(
    session_id: str = Form(...),
    day_key: str = Form(...),
    ordinal: int = Form(...),
    session_started_at: str = Form(...),
    course_id: str = Form(_DEFAULT_COURSE_ID),
) -> dict:
    """세션 구분선 옆 삭제 버튼(2026-10-08) -- 이 session_id에 속한 필기를
    DB에서 통째로 지운다. 확인(confirm) 다이얼로그는 프론트엔드가 담당하고,
    여기선 실제 삭제만 수행한다.

    day_key/ordinal/session_started_at은 프론트엔드가 그 구분선을 그릴 때
    이미 계산해둔 값을 그대로 넘겨받는다 -- 필기 행이 다 지워진 뒤에는 "몇
    번째 세션이었는지"를 note_chunks만으로는 더 이상 알 수 없으므로, 삭제
    직전에 이 값들을 묘비(deleted_sessions)로 같이 남겨서 재구성 시 번호가
    안 꼬이게 한다(사용자 요청, CLAUDE.md 참고).
    """
    deleted = delete_session_notes(session_id)
    insert_deleted_session(session_id, course_id, day_key, ordinal, session_started_at)
    return {"ok": True, "deleted": deleted}


@app.get("/api/deleted_sessions")
def api_deleted_sessions(course_id: str = _DEFAULT_COURSE_ID) -> list[dict]:
    """필기 타임라인에 "N번째 필기세션 삭제됨" 구분선을 끼워 넣기 위한 묘비 목록."""
    rows = get_deleted_sessions(course_id)
    return [
        {
            "day_key": r["day_key"],
            "ordinal": r["ordinal"],
            "session_started_at": r["session_started_at"].isoformat(),
        }
        for r in rows
    ]


@app.post("/api/end_session")
def api_end_session(session_id: str = Form(...)) -> dict:
    """"필기 종료" 버튼이 호출 -- 세션 내 작업기억(InMemory)을 폐기하고
    session_events에 session_ended를 기록한다(src/session_memory.py).
    서버 프로세스는 그대로 두고(아직 다른 과목/세션으로 계속 쓸 수 있으니),
    종료는 /api/shutdown이 별도로 담당한다.
    """
    end_session(session_id)
    return {"ok": True}


@app.post("/api/shutdown")
def api_shutdown() -> dict:
    """"필기 종료" 버튼이 세션 종료 다음으로 호출 -- 서버 프로세스 자체를 내린다.

    로컬 1인용 서비스라 HTTP 요청으로 서버를 내리는 걸 허용한다(2026-10-08,
    [[deferred-launch-exit-ux]] -- 여러 사용자가 쓰는 서비스였다면 위험한
    패턴이라 하지 않았을 것). 이 요청의 응답이 먼저 클라이언트에 전달돼야
    하므로, 핸들러 안에서 바로 종료하지 않고 아주 짧게 지연시킨 뒤
    `os._exit()`로 즉시 종료한다 -- 로컬 dev 서버라 정상적인 연결 드레이닝
    없이 바로 꺼져도 문제없음.
    """
    threading.Timer(0.3, lambda: os._exit(0)).start()
    return {"ok": True}

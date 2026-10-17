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
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

# uvicorn --reload는 서브프로세스로 뜨는데, 그 프로세스의 작업 디렉터리 기준
# 자동 탐색이 안 먹힐 때가 있어서(2026-10-02 실측) 절대경로로 명시한다.
load_dotenv(dotenv_path=str(Path(__file__).resolve().parent.parent / ".env"))

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from src.api import ingest_file, ingest_note, parse_file
from src.db import (
    get_all_notes,
    get_deleted_sessions,
    get_recent_session_boundary,
    get_session_time_range,
    get_today_hints,
    get_upload_labels,
    get_hint_excluded_origin_refs,
    insert_deleted_session,
    list_sessions,
    record_hint_exclusion,
    record_hint_feedback,
    record_line_deletion,
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
# dry_run(test:: 태그) 입력의 생성 결과 기록 (2026-10-10 결정) -- note_chunks/
# 세션 작업기억/힌트 노출 통계 중 어디에도 안 남기는 게 "RAG 오염 방지"의
# 원래 의도였는데, 그러다 보니 테스트 결과를 나중에 다시 확인할 방법이 전혀
# 없어져버림(사용자 지적). note_chunks와 완전히 분리된 이 파일에만 남겨서 --
# 검색/임베딩/통계 어디에도 안 쓰이니(RAG 소스 아님) 오염 없이, 그래도
# 생성된 힌트 내용은 나중에 검토할 수 있게 한다.
_TEST_RUN_LOG = _PROJECT_ROOT / "data" / "test_run_log.jsonl"
# "피드백선택평가셋"(/review/5v6, 2026-10-17) 전용 -- 5차/6차 힌트 중 어느 쪽을
# 더 선호하는지(A/B 중 하나, 상호배타) 기록. 기존 +/-/✓(_HINT_FEEDBACK_LOG)는
# 힌트 하나하나에 대한 개별 평가라 그대로 두고, 이건 "둘 중 어느 쪽이 나은가"라는
# 별개 성격의 비교 데이터라 별도 로그로 분리(note_chunks에도 안 씀 -- 순수
# 모델 비교/의사결정용 데이터).
_MODEL_PREFERENCE_LOG = _PROJECT_ROOT / "data" / "model_preference_log.jsonl"

app = FastAPI(title="NoteMate API")

# 파괴적 엔드포인트(/api/delete_session, /api/shutdown) 전용 최소 방어선
# (2026-10-09, 외부 리뷰 Q2 반영). 로컬 1인용 서비스라 정식 인증/토큰은
# 발표 이후 과제로 미루고, 지금은 비용이 거의 없는 두 가지만 확인한다.
_ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1"}
_ALLOWED_ORIGINS = {"http://127.0.0.1:8000", "http://localhost:8000"}


def _require_local_origin(request: Request) -> None:
    """1) Host 헤더가 로컬이 아니면 거부(DNS 리바인딩 방어) -- 공격자 도메인이
    처음엔 자기 서버를 가리키다 DNS TTL이 지나면 127.0.0.1로 바뀌는 공격을
    막는다. 2) Origin 헤더가 있는데 이 서비스 자신이 아니면 거부 -- 같은
    PC에서 열려 있는 다른(악성/손상된) 탭이 이 로컬 서버로 blind POST를
    보내는 걸 막는다. 둘 다 IP(`request.client.host`) 체크만으로는 못
    막는 경로라는 게 외부 리뷰의 핵심 지적이었음(브라우저를 거친 요청은
    항상 127.0.0.1에서 나가서 IP 검사를 그냥 통과함).

    `Depends()`로 라우트에 걸어야 한다 -- `@app.middleware("http")` 안에서
    `HTTPException`을 raise하면 FastAPI의 예외 처리 경로 바깥이라 403이
    아니라 500으로 나가는 함정이 있다(외부 리뷰 지적, 실제로 맞는 내용).
    """
    host = (request.url.hostname or "").lower()
    if host not in _ALLOWED_HOSTS:
        raise HTTPException(status_code=403, detail="forbidden host")
    origin = request.headers.get("origin")
    if origin and origin not in _ALLOWED_ORIGINS:
        raise HTTPException(status_code=403, detail="forbidden origin")


def _append_hint_event(record: dict) -> None:
    _HINT_FEEDBACK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(_HINT_FEEDBACK_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _append_test_event(record: dict) -> None:
    _TEST_RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(_TEST_RUN_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _append_preference_event(record: dict) -> None:
    _MODEL_PREFERENCE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(_MODEL_PREFERENCE_LOG, "a", encoding="utf-8") as f:
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


@app.get("/review/5v6")
def serve_5v6_review() -> FileResponse:
    """5차(4.1-mini) vs 6차(gpt-6-luna) 힌트 비교 "피드백선택평가셋"(2026-10-17).

    /review/run3과 같은 패턴(같은 출처라 CORS 없이 /api/hint_feedback 직접 호출) --
    `scripts/generate_5v6_review.py`가 생성한 정적 파일을 그대로 서빙만 한다.
    (10/17에 사용자가 짚은 구분: 이건 "피드백선택평가셋"이고, run3 쪽은
    "Tier1 적절성 블라인드 평가셋"이라 성격이 다름 -- 둘 다 별도로 필요.)
    """
    return FileResponse(
        _PROJECT_ROOT / "data" / "review_5v6_hint_comparison.html",
        headers=_NO_CACHE_HEADERS,
    )


@app.post("/api/ingest_note")
def api_ingest_note(
    # Form(...)(필수)로 두면 빈 문자열("")이 넘어올 때 FastAPI가 이를
    # None으로 취급해 "필드 없음" 422를 내는 버그가 있었다(2026-10-10 실측,
    # curl/urllib 둘 다로 재현 확인) -- 줄 삭제(빈 내용 제출)를 지원하려면
    # 빈 문자열이 정상적으로 통과해야 해서 명시적 기본값(Form(""))으로 바꿈.
    content: str = Form(""),
    course_id: str = Form(_DEFAULT_COURSE_ID),
    search_course_ids: str | None = Form(None),
    session_id: str = Form(...),
    origin_prefix: str = Form("live"),
    client_line_id: str | None = Form(None),
    dry_run: bool = Form(False),
    inserted_out_of_place: bool = Form(False),
    allow_general_knowledge: bool = Form(True),
    medium: str = Form("txt"),
    upload_label: str | None = Form(None),
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

    `dry_run`: True면 검색/판단/힌트 생성은 기존 데이터 기준으로 실제로
    돌아가지만 저장은 안 한다(2026-10-10) -- notemate_mockup.html의
    `test::` 테스트 입력이 쓴다. note_chunks/세션 작업기억/힌트 노출 통계
    어디에도 안 남기는 건 그대로지만(RAG 오염 방지 취지 유지), 결과 자체를
    나중에 확인할 수 있게 `_TEST_RUN_LOG`에만 따로 남긴다(사용자 지적: 테스트
    결과를 재확인할 방법이 전혀 없었음).

    `inserted_out_of_place`: 사용자가 타임라인 중 과거 세션/날짜 블록으로
    스크롤해서 그 자리에 입력한 경우 True(2026-10-10). 판단/라우팅엔 안
    쓰이고 그대로 저장만 됨 -- 화면에서 다른 글자색으로 표시하는 용도
    ([[deferred-historical-note-insertion]], 제대로 된 "실제 그 자리에
    소속"은 차후과제).

    `allow_general_knowledge`: False면 Tier2가 근거(강의자료/내 필기)를 하나도
    못 찾았을 때 힌트 생성 자체를 건너뛴다(2026-10-12, src/nodes.py
    tier2_deep_agent 참고) -- 힌트 남발 완화 + 비교테스트용. 기본값 True는
    기존 동작 그대로.

    `is_ask`는 별도 Form 필드 없이 `origin_prefix == "ask"`로 자동 판별해서
    넘김(2026-10-12) -- 근거 부족 + 바로 물어보기일 때만 웹검색(Tavily)이
    개입하도록 src/nodes.py tier2_deep_agent가 이 값을 씀.

    `medium`/`upload_label`(2026-10-17, 재생모드 업로드): 기본값은 기존 동작과
    동일(txt/None). 재생모드 프론트엔드 루프가 `origin_prefix="replay"` +
    `/api/upload_for_replay`가 돌려준 `medium`(파일 형식에 따라 txt/pdf/notion
    등)을 그대로 실어 보낸다 -- 라이브 타이핑 경로는 건드리지 않음.

    `search_course_ids`(2026-10-17 야간, 노트 묶음/RAG 공유그룹): 콤마로 구분된
    course_id 목록 문자열(예: "a,b,c"). 프론트가 지금 노트가 속한 묶음 전체를
    계산해서 보낸다 -- 안 보내면(None) `ingest_note()`가 `course_id` 하나로
    대체해서 기존 동작과 동일. 저장되는 노트(`course_id`)는 안 바뀌고, Tier2
    검색/확신도 계산의 "범위"만 넓어진다.

    content가 빈 문자열(필기 줄을 지우고 제출한 경우): 전체 파이프라인을
    타지 않고 `record_line_deletion()`으로 바로 간다(2026-10-10, 사용자
    요청 -- "수정이 되는데 삭제가 안 되는 건 이상하다"). `client_line_id`가
    없으면(한 번도 저장 안 된, 방금 만들었다 바로 지운 줄) 지울 것 자체가
    없으니 완전히 조용히 끝낸다.
    """
    if not content.strip():
        if client_line_id:
            origin_ref = f"{origin_prefix}:{session_id}:{datetime.now().isoformat()}:deleted"
            record_line_deletion(client_line_id, course_id, session_id, origin_ref)
        return {
            "routing_decision": None,
            "top_score": None,
            "is_question": None,
            "tier2_result": None,
            "tier2_sources": None,
            "origin_ref": None,
        }

    origin_ref = f"{origin_prefix}:{session_id}:{datetime.now().isoformat()}"
    parsed_search_course_ids = (
        [c.strip() for c in search_course_ids.split(",") if c.strip()] if search_course_ids else None
    )
    state = ingest_note(
        content=content,
        source="personal",
        medium=medium,
        origin_ref=origin_ref,
        course_id=course_id,
        search_course_ids=parsed_search_course_ids,
        timestamp=datetime.now(),
        session_id=session_id,
        client_line_id=client_line_id,
        dry_run=dry_run,
        inserted_out_of_place=inserted_out_of_place,
        allow_general_knowledge=allow_general_knowledge,
        is_ask=(origin_prefix == "ask"),
        upload_label=upload_label,
    )
    if dry_run:
        # note_chunks/세션 작업기억/힌트 노출 통계와 완전히 분리된 파일 --
        # RAG 검색이나 실사용 지표 어디에도 안 쓰이니, 내용만 나중에 검토 가능.
        _append_test_event({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "course_id": course_id,
            "session_id": session_id,
            "content": content,
            "routing_decision": state.get("routing_decision"),
            "top_score": state.get("top_score"),
            "tier2_result": state.get("tier2_result"),
            "tier2_sources": state.get("tier2_sources"),
        })
    elif state.get("tier2_result"):
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
        # 2026-10-11 수정: assess_note가 판단 결과를 state["assessment"]["is_question"]
        # (중첩)에 저장하는데, 여긴 state.get("is_question")(평평한 키)을 읽고
        # 있어서 그런 키가 state에 존재한 적이 없어 항상 None을 반환하던 버그였음
        # (백엔드 베타테스트 중 발견 -- 질문형 문장을 넣어도 응답이 항상
        # is_question=null로 와서 역추적함). 라우팅 자체는 내부에서
        # state["assessment"]를 그대로 쓰므로 영향 없었고, 이 API 응답 필드만
        # 항상 비어있었음.
        "is_question": (state.get("assessment") or {}).get("is_question"),
        "tier2_result": state.get("tier2_result"),
        # 2026-10-09: 이 힌트가 뭘 근거로 만들어졌는지(내 필기/강의자료/일반
        # 지식) -- 강사 피드백 + 사용자 질문 반영, src/nodes.py 참고.
        "tier2_sources": state.get("tier2_sources"),
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
    """힌트(hint) 또는 바로 물어보기 답변(ask)의 버튼(+ / − / ✓) 클릭을 기록한다.

    `✓`는 compute_routing_decision()의 라우팅 억제(Phase 2, 2026-10-08)에 쓰이고,
    `+`/`−`는 tier2_deep_agent()의 힌트 생성 스타일/억제(2026-10-17)에 쓰인다 --
    "라우팅에는 연결 안 함"이라던 이전 문구는 ✓ 억제가 추가되며 틀린 서술이
    됐었음(수정됨).

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


@app.post("/api/hint_exclude")
def api_hint_exclude(
    origin_ref: str = Form(...),
    excluded: bool = Form(...),
) -> dict:
    """"힌트 예외처리" 체크박스(2026-10-17 야간 설계, `+`/`-`/`✓`와는 별개 신호) --

    이 줄은 양식/템플릿이라 힌트 생성 대상 자체가 아니라고 표시(또는 해제)한다.
    `feedback_value`를 공유하지 않고 `note_chunks.hint_excluded` 전용 컬럼에
    기록 -- 품질 반응(+/-/✓)과 범주 제외(이거)가 같은 줄에 동시에 존재할 수
    있어야 하기 때문. 지금은 이 신호가 쌓이기만 하고(평가셋/피드백선택평가셋
    진행용) 실제 라우팅/힌트생성 억제 로직에는 아직 연결 안 함 -- 그건 서비스
    화면(notemate_mockup.html)의 확인/복구 UI와 함께 이후 별도 작업.
    """
    reflected = record_hint_exclusion(origin_ref, excluded)
    return {"ok": True, "reflected": reflected}


@app.get("/api/hint_exclude_state")
def api_hint_exclude_state() -> dict:
    """현재 예외처리(hint_excluded=TRUE)된 origin_ref 목록을 {origin_ref: true} 형태로 반환.

    새로고침 후 체크박스 선택 상태를 복원하는 데 쓴다(hint_feedback_state와 같은 용도).
    """
    return {ref: True for ref in get_hint_excluded_origin_refs()}


@app.post("/api/model_preference")
def api_model_preference(
    session_id: str = Form(...),
    pair_key: str = Form(...),
    preferred_origin: str = Form(""),
) -> dict:
    """"피드백선택평가셋"(/review/5v6) 전용 -- 한 줄의 힌트 A/B 중 어느 쪽이
    더 나은지(상호배타 선택) 기록. `pair_key`는 두 origin_ref를 정렬해서
    합친 값이라 A/B 배치(화면 표시 순서, 줄마다 랜덤)와 무관하게 항상
    같은 쌍을 가리킨다. `preferred_origin`을 비워 보내면 "선택 해제"
    (다시 클릭해서 토글한 경우).
    """
    _append_preference_event({
        "ts": datetime.now().isoformat(timespec="seconds"),
        "session_id": session_id,
        "pair_key": pair_key,
        "preferred_origin": preferred_origin or None,
    })
    return {"ok": True}


@app.get("/api/model_preference_state")
def api_model_preference_state() -> dict:
    """pair_key별 마지막 선택(preferred_origin)만 뽑아 돌려준다 -- 새로고침
    후 선택 표시 복원용(/api/hint_feedback_state와 같은 패턴)."""
    state: dict[str, str | None] = {}
    if _MODEL_PREFERENCE_LOG.exists():
        with open(_MODEL_PREFERENCE_LOG, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                key = record.get("pair_key")
                if key:
                    state[key] = record.get("preferred_origin")
    return state


@app.post("/api/ingest_file")
async def api_ingest_file(
    file: UploadFile = File(...),
    source: str = Form("base"),
    course_id: str = Form(_DEFAULT_COURSE_ID),
    upload_label: str | None = Form(None),
) -> dict:
    """업로드 버튼이 호출 -- 받은 파일을 data/<source>/에 저장한 뒤 ingest_file()로 처리.

    data/<source>/ 아래에 실물 파일로 남겨두는 이유: 기존 `ingest_folder.py`
    폴더 컨벤션과 그대로 호환되고(나중에 일괄 재처리/검증 스크립트가 그대로
    먹힘), 원본을 디스크에 보존해 문제 생기면 재인제스트 가능.

    `upload_label`: 업로드 UI의 분류명 드롭다운(과목 선택과 같은 UX, 2026-10-12)
    -- 판단/라우팅엔 안 쓰이고 note_chunks.upload_label 컬럼에 그대로 저장만 됨.
    """
    dest_dir = _PROJECT_ROOT / "data" / source
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / file.filename

    with open(dest_path, "wb") as f:
        f.write(await file.read())

    results = ingest_file(str(dest_path), source=source, course_id=course_id, upload_label=upload_label)
    return {"chunks_saved": len(results), "filename": file.filename}


@app.post("/api/upload_for_replay")
async def api_upload_for_replay(
    file: UploadFile = File(...),
    source: str = Form("base"),
) -> dict:
    """"재생모드로 보여주기" 업로드 전용 -- 파일을 저장하고 **파싱만** 해서
    조각 목록을 돌려준다 (2026-10-17, [[deferred-visual-replay-upload-feature]]).

    `/api/ingest_file`과 달리 ingest_note()를 호출하지 않는다 -- 그 호출은
    프론트엔드가 받은 조각 목록을 1~2초 간격으로 하나씩 `/api/ingest_note`
    (origin_prefix="replay")에 넘기면서 화면에 실시간으로 재생하는 쪽에서
    담당한다. 파일 형식(txt/pdf/notion)과 무관하게 동일하게 동작 -- 각
    어댑터가 이미 내놓는 청크 리스트(줄/문단/페이지 단위)를 그대로 쓸 뿐,
    재생을 위한 별도 파싱 단계가 필요 없다(사용자 지적으로 설계 수정됨).
    """
    dest_dir = _PROJECT_ROOT / "data" / source
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / file.filename

    with open(dest_path, "wb") as f:
        f.write(await file.read())

    notes, medium = parse_file(str(dest_path))
    return {
        "filename": file.filename,
        "medium": medium,
        "chunks": [
            {"content": n.content, "timestamp": n.timestamp.isoformat() if n.timestamp else None}
            for n in notes
        ],
    }


@app.get("/api/upload_labels")
def api_upload_labels() -> list[str]:
    """업로드 분류명 드롭다운이 열릴 때 호출 -- 지금까지 실제로 쓰인 분류명
    전부를 돌려준다(2026-10-12). 프론트엔드가 이걸 localStorage 목록과 합쳐서
    보여줘서, 기기/브라우저가 달라도 기존에 쓴 분류명을 그대로 재사용할 수
    있게 함(사용자 지적: "기존 메타데이터 값을 통일시켜 주는 게 좋다")."""
    return get_upload_labels()


@app.get("/api/today_hints")
def api_today_hints(course_id: str = _DEFAULT_COURSE_ID) -> list[dict]:
    """오른쪽 "오늘의 요약" 패널용 -- 오늘 Tier2가 개입한 힌트 목록."""
    hints = get_today_hints(course_id)
    return [
        {
            "content": h["content"],
            "tier2_result": h["tier2_result"],
            "tier2_sources": h["tier2_sources"],
            "created_at": h["created_at"].isoformat(),
        }
        for h in hints
    ]


@app.get("/api/recent_session_boundary")
def api_recent_session_boundary(
    course_id: str = _DEFAULT_COURSE_ID,
    session_limit: int = 10,
) -> dict:
    """서비스 시작 분할 로딩의 경계 시각 -- 프론트엔드가 로딩 시작 때 제일
    먼저 호출한다. 이 시각 이후(`after`)만 먼저 불러와 빠르게 화면을 띄우고,
    이전(`before`)은 백그라운드에서 조용히 불러와 맨 위에 끼워 넣는다.

    boundary가 null이면 "자를 필요 없이 전부 로딩"하라는 뜻 -- 히스토리가
    작은 과목은 분할 자체가 무의미하므로 프론트엔드가 기존 동작(전체 한 번에
    로딩)으로 자연히 돌아간다."""
    boundary = get_recent_session_boundary(course_id, session_limit)
    return {"boundary": boundary.isoformat() if boundary else None}


@app.get("/api/all_notes")
def api_all_notes(
    course_id: str = _DEFAULT_COURSE_ID,
    after: str | None = None,
    before: str | None = None,
) -> list[dict]:
    """페이지를 새로고침하거나 서비스를 재시작해도 이 과목의 필기 전체가 그대로
    복원되도록(2026-10-08, "종료 후 재시작 시 기존 필기가 떠있어야 한다"는
    요청 반영 -- 구 이름 api_today_notes, 날짜 제한 제거) -- 가짜 데모 문단
    대신 입력된 personal 필기 전체(Tier1/2 가리지 않고)를 반환.

    `after`/`before`(2026-10-14 추가, ISO 타임스탬프): 서비스 시작 속도
    개선을 위한 분할 로딩에 쓰임 -- 최근분(`after`)은 빠르게 먼저 불러와
    화면을 띄우고, 그 전 과거분(`before`)은 프론트엔드가 백그라운드에서
    따로 불러와 조용히 맨 위에 끼워 넣는다(`/api/recent_session_boundary`
    참고). 둘 다 안 주면 기존과 동일하게 전체를 반환."""
    after_dt = datetime.fromisoformat(after) if after else None
    before_dt = datetime.fromisoformat(before) if before else None
    notes = get_all_notes(course_id, after=after_dt, before=before_dt)
    return [
        {
            "content": n["content"],
            "tier2_result": n["tier2_result"],
            "tier2_sources": n["tier2_sources"],
            "created_at": n["created_at"].isoformat(),
            "client_line_id": n["client_line_id"],
            "origin_ref": n["origin_ref"],
            "session_id": n["session_id"],
            "inserted_out_of_place": n["inserted_out_of_place"],
        }
        for n in notes
    ]


_KST = ZoneInfo("Asia/Seoul")


@app.get("/api/export_notes")
def api_export_notes(
    course_id: str = _DEFAULT_COURSE_ID,
    start: str | None = None,
    end: str | None = None,
    session_ids: str | None = None,
) -> list[dict]:
    """노트 내보내기(2026-10-18) -- 조건에 맞는 원본 데이터를 반환하고,
    실제 문서/텍스트 조립은 프론트엔드가 담당한다(화면에 이미 있는 날짜
    구분 로직을 그대로 재사용하기 위해 -- 같은 걸 서버에서 또 구현하면
    화면 표시와 내보내기 결과가 서로 어긋날 위험이 생김).

    `start`/`end`는 KST 달력 날짜(YYYY-MM-DD) 문자열, 둘 다 생략하면 전체
    기간(지금 보이는 화면과 동일). `get_all_notes()`에는 "after만 있으면
    sim: 데이터를 일부러 제외하는 빠른 경로"가 있는데(실시간 분할로딩
    최적화용, 위 api_all_notes 참고) -- 내보내기에서 그 경로를 타면 "최근
    날짜부터" 내보낼 때 재생 테스트 데이터가 조용히 빠지는 사고가 날 수
    있다. 그래서 `after`를 줄 땐 항상 `before`도 같이 줘서(끝 날짜 생략
    시 지금(KST)+1일을 기본값으로) 항상 완전한(느리지만 안전한) 경로를
    타도록 강제한다.

    `session_ids`(콤마구분, `start`/`end`와 배타적-- 프론트가 둘 중 하나만
    보냄): 날짜범위의 한계(재생 테스트로 대량 백필된 데이터는 실제 역사적
    날짜가 `created_at`에 안 남아있어서 날짜범위로 못 골라짐, 사용자와 논의
    후 결정) 보완책 -- 날짜와 무관하게 세션 단위로 직접 골라서 내보낼 수
    있게 한다. 이 경우 전체를 가져온 뒤 파이썬에서 필터링한다(내보내기는
    자주 발생하는 핫패스가 아니라 단순함을 우선).
    """
    if session_ids:
        wanted = set(session_ids.split(","))
        notes = [n for n in get_all_notes(course_id) if n["session_id"] in wanted]
    else:
        after_dt = datetime.strptime(start, "%Y-%m-%d").replace(tzinfo=_KST) if start else None
        before_dt = (
            datetime.strptime(end, "%Y-%m-%d").replace(tzinfo=_KST) + timedelta(days=1) if end else None
        )
        if after_dt is not None and before_dt is None:
            before_dt = datetime.now(_KST) + timedelta(days=1)
        notes = get_all_notes(course_id, after=after_dt, before=before_dt)
    return [
        {
            "content": n["content"],
            "tier2_result": n["tier2_result"],
            "tier2_sources": n["tier2_sources"],
            "created_at": n["created_at"].isoformat(),
            "origin_ref": n["origin_ref"],
            "session_id": n["session_id"],
        }
        for n in notes
    ]


@app.get("/api/sessions")
def api_sessions(course_id: str = _DEFAULT_COURSE_ID) -> list[dict]:
    """노트 내보내기 "세션 선택" 모드의 선택 목록용 -- session_id/시작시각/줄수.

    라벨(화면의 "N번째 필기" 같은 서수)까지는 안 만든다 -- 그건 전체
    타임라인을 순서대로 훑어야 계산되는 화면 전용 로직이라(notemate_mockup.html
    참고), 여기선 시작시각+줄수만 주고 "YYYY-MM-DD HH:MM · N줄"처럼 프론트가
    가볍게 표시하면 충분하다.
    """
    sessions = list_sessions(course_id)
    return [
        {
            "session_id": s["session_id"],
            "started_at": s["started_at"].isoformat(),
            "line_count": s["line_count"],
        }
        for s in sessions
    ]


@app.post("/api/day_summary")
def api_day_summary(period_key: str = Form(...), course_id: str = Form(_DEFAULT_COURSE_ID)) -> dict:
    """필기 타임라인의 날짜 구분선 "지난 시간 복습" 버튼이 호출.

    period_key: YYYYMMDD. 같은 날짜를 또 요청하면 재생성 없이 캐시된 요약을
    그대로 반환(src/summarize.py 참고).
    """
    return generate_day_summary(course_id, period_key)


@app.post("/api/delete_session", dependencies=[Depends(_require_local_origin)])
def api_delete_session(
    session_id: str = Form(...),
    ordinal: int = Form(...),
    course_id: str = Form(_DEFAULT_COURSE_ID),
) -> dict:
    """세션 구분선 옆 삭제 버튼(2026-10-08 신설, 2026-10-09 설계 변경).

    **더 이상 note_chunks를 실제로 지우지 않는다** -- 사용자 결정: "그것도
    데이터인데 아예 복구 못 하고 기록도 못 보게 없애는 건 아닌 것 같다.
    그 경우의 수도 많지 않을 거고(보통 하루에 과목당 세션 하나)". 화면/
    새로고침 재구성에서 숨기는 역할은 `deleted_sessions`에 올리는 것만으로
    충분하다(`get_all_notes()`가 거기 올라간 session_id를 걸러냄).

    `day_key`/`session_started_at`은 더 이상 프론트가 보내지 않는다 -- 외부
    리뷰(Claude웹+Gemini) 지적: "그 값들은 서버가 원본 행에서 직접 도출할
    수 있으니 클라이언트를 신뢰하지 말라". `get_session_time_range()`로
    그 세션의 가장 이른 필기 시각을 서버가 직접 구하고, KST로 변환해서
    `day_key`를 만든다 -- UTC 그대로 쓰면 자정~오전9시 세션이 전날로 밀리는
    버그가 생긴다(같은 리뷰 지적). 세션에 필기가 하나도 없으면(만들자마자
    지운 빈 세션) 묘비를 남기지 않는다 -- 시작 시각을 현재 시각으로
    날조하면 묘비가 엉뚱한 위치에 나타나는 조용한 데이터 오류가 되므로.

    `ordinal`만은 "생성 시점에 클라이언트가 매긴 순번"이라 서버가 대신 구할
    방법이 없어 그대로 신뢰한다(양쪽 리뷰 공통 결론, 레이스 가능성은
    로컬 1인용 규모에서 무시 가능하다고 판단).
    """
    try:
        uuid.UUID(session_id)
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=400, detail="invalid session_id")

    started_at = get_session_time_range(session_id)
    if started_at is None:
        return {"ok": True, "tombstoned": False}

    day_key = started_at.astimezone(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d")
    insert_deleted_session(session_id, course_id, day_key, ordinal, started_at.isoformat())
    # 세션 내 작업기억(인메모리, src/session_memory.py)도 같이 비운다
    # (2026-10-10, 외부 리뷰 지적) -- 안 비우면, 지운 세션의 session_id로
    # (드물지만, 예: 지운 세션이 지금 떠 있는 라이브 세션이었던 경우) 계속
    # 필기를 이어 쓸 때 "세션 내 중복 억제" 판단이 이미 지워진 내용까지
    # 근거로 계속 쓰게 된다. end_session()은 없는 session_id를 넘겨도
    # 조용히 무시하므로, 이미 세션이 끝나있던 경우에도 안전하게 호출 가능.
    end_session(session_id)
    return {"ok": True, "tombstoned": True}


@app.get("/api/deleted_sessions")
def api_deleted_sessions(
    course_id: str = _DEFAULT_COURSE_ID,
    after: str | None = None,
    before: str | None = None,
) -> list[dict]:
    """필기 타임라인에 "N번째 필기세션 삭제됨" 구분선을 끼워 넣기 위한 묘비 목록.

    `after`/`before`(2026-10-14): /api/all_notes와 같은 분할 로딩 경계를
    묘비 쪽에도 동일하게 적용."""
    after_dt = datetime.fromisoformat(after) if after else None
    before_dt = datetime.fromisoformat(before) if before else None
    rows = get_deleted_sessions(course_id, after=after_dt, before=before_dt)
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


@app.post("/api/shutdown", dependencies=[Depends(_require_local_origin)])
def api_shutdown() -> dict:
    """"필기 종료" 버튼이 세션 종료 다음으로 호출 -- 서버 프로세스 자체를 내린다.

    로컬 1인용 서비스라 HTTP 요청으로 서버를 내리는 걸 허용한다(2026-10-08,
    [[deferred-launch-exit-ux]] -- 여러 사용자가 쓰는 서비스였다면 위험한
    패턴이라 하지 않았을 것). Origin/Host 검증은 2026-10-09 외부 리뷰 반영
    (`_require_local_origin` 참고) -- 같은 PC의 다른 탭이 blind POST로
    서버를 끄는 것까지는 막아야 한다고 판단. 이 요청의 응답이 먼저
    클라이언트에 전달돼야 하므로, 핸들러 안에서 바로 종료하지 않고 아주
    짧게 지연시킨 뒤 `os._exit()`로 즉시 종료한다 -- 로컬 dev 서버라 정상적인
    연결 드레이닝
    없이 바로 꺼져도 문제없음.
    """
    threading.Timer(0.3, lambda: os._exit(0)).start()
    return {"ok": True}

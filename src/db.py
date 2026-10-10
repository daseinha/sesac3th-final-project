"""Supabase(Postgres) 연결 및 note_chunks 쿼리 헬퍼.

supabase-py(REST API) 대신 psycopg로 직접 연결한다 (CLAUDE.md 2026-09-29
결정 참고 -- 복합 SQL 쿼리 자유도 확보 목적). 커넥션 풀은 프로세스당 하나만
만들어 재사용한다.
"""

import os
import re
from datetime import datetime

from psycopg_pool import ConnectionPool

_pool: ConnectionPool | None = None


def get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        _pool = ConnectionPool(
            conninfo=os.environ["POSTGRES_URI"],
            min_size=1,
            max_size=5,
            open=True,
        )
    return _pool


# 삭제된 세션(묘비, deleted_sessions)의 내용을 note_chunks를 읽는 모든 경로에서
# 제외하기 위한 공통 SQL 조각(2026-10-10, 외부 리뷰 Q1 반영) -- 처음엔
# get_all_notes()(화면 재구성)에만 넣었었는데, 검색/판단 쪽(이 아래 네 함수)엔
# 전혀 안 걸러지고 있던 걸 리뷰에서 지적받아 실제로 확인함. 그래서 소프트
# 삭제했는데도 그 내용이 라우팅 확신도(top_score)·Tier2 근거·"오늘의 요약"에
# 계속 영향을 줄 수 있었음. `NOT IN` 대신 `NOT EXISTS`를 쓴다 -- `NOT IN`은
# 서브쿼리 결과에 NULL이 하나라도 섞이면 전체가 사라지는 함정이 있는데(지금은
# `deleted_sessions.session_id`가 PK라 NULL이 될 수 없어 실제로는 안전하지만),
# `NOT EXISTS`는 그 함정 자체가 구조적으로 없어서 더 안전한 기본값으로 택함.
# `note_chunks.session_id`가 NULL인 행(이 컬럼이 생기기 전 과거 데이터)은
# 비교가 "불명(unknown)"이 되어 자동으로 매치 안 되니 따로 `IS NULL OR`를
# 붙일 필요도 없음.
_ACTIVE_SESSION_FILTER = """
    AND NOT EXISTS (
        SELECT 1 FROM deleted_sessions d WHERE d.session_id = note_chunks.session_id
    )
"""


def search_top_similarity(embedding: list[float], course_id: str | None) -> dict | None:
    """note_chunks에서 embedding과 가장 유사한 기존 행의 코사인 유사도(0~1에 가까움)와
    그 행의 피드백 상태(feedback_value/feedback_at)를 함께 반환.

    비교할 기존 행이 하나도 없으면(테이블이 비어있는 등) None을 반환한다 --
    "지원 자료가 없다"는 의미이므로 호출부(assess_note)에서 확신도 낮음으로 처리한다.

    2026-10-08: 피드백 반영 구조(✓ 억제) 추가 -- 가장 비슷한 과거 필기가 "이미
    알고 있음(✓)" 피드백을 받았었는지를 assess_note가 같이 알아야 해서, 이
    top-1 조회에 얹어서 같이 가져온다(쿼리 한 번 더 안 날리려고).

    파이썬 list를 psycopg에 그냥 넘기면 double precision[]로 직렬화돼서
    pgvector의 <=> 연산자가 못 알아본다 -- 그래서 %s::vector로 명시적으로
    캐스팅한다 (register_vector 어댑터 자동 등록보다 이 방식이 더 안정적).
    """
    pool = get_pool()
    course_filter = "AND course_id = %(course_id)s" if course_id else ""
    query = f"""
        SELECT 1 - (embedding <=> %(embedding)s::vector) AS similarity,
               feedback_value, feedback_at
        FROM note_chunks
        WHERE embedding IS NOT NULL
        {course_filter}
        {_ACTIVE_SESSION_FILTER}
        ORDER BY embedding <=> %(embedding)s::vector
        LIMIT 1;
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, {"embedding": embedding, "course_id": course_id})
            row = cur.fetchone()
            if not row:
                return None
            return {"similarity": row[0], "feedback_value": row[1], "feedback_at": row[2]}


def record_hint_feedback(origin_ref: str, value: str) -> bool:
    """이 힌트가 달린 원본 필기 행(note_chunks)에 피드백을 기록한다.

    기존 feedback_value가 있어도 마지막 선택으로 덮어쓴다(HINT_FEEDBACK_SPEC.md
    "같은 힌트에 여러 번 누르면 마지막 선택을 기준으로 한다" 원칙 그대로).
    origin_ref가 UNIQUE라 최대 한 행만 바뀐다. 매칭되는 행이 없으면 False.
    """
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE note_chunks SET feedback_value = %(value)s, feedback_at = now() "
                "WHERE origin_ref = %(origin_ref)s;",
                {"value": value, "origin_ref": origin_ref},
            )
            conn.commit()
            return cur.rowcount > 0


def search_similar_chunks(
    embedding: list[float], course_id: str | None, k: int = 5
) -> list[dict]:
    """note_chunks에서 embedding과 유사한 상위 k개 행을 반환 (content/source/origin_ref 포함).

    assess_note의 search_top_similarity()는 점수 하나만 필요해서 top-1만 보지만,
    Tier2 딥 에이전트는 실제 근거로 LLM에 넘길 내용 자체가 필요해서 여러 개를 가져온다.
    """
    pool = get_pool()
    course_filter = "AND course_id = %(course_id)s" if course_id else ""
    query = f"""
        SELECT content, source, origin_ref,
               1 - (embedding <=> %(embedding)s::vector) AS similarity
        FROM note_chunks
        WHERE embedding IS NOT NULL
        {course_filter}
        {_ACTIVE_SESSION_FILTER}
        ORDER BY embedding <=> %(embedding)s::vector
        LIMIT %(k)s;
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, {"embedding": embedding, "course_id": course_id, "k": k})
            columns = [c.name for c in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]


def fetch_corpus_for_bm25(course_id: str | None) -> list[dict]:
    """BM25 인덱스를 만들기 위해 course_id 범위의 note_chunks content를 전부 가져온다.

    LangChain의 BM25Retriever가 인메모리 방식이라, 매 검색 시점에 이 함수로
    코퍼스를 불러와서 즉석으로 인덱스를 만든다(src/retrieval.py) -- 캐시가
    아예 없으니 삭제된 세션을 걸러내는 것도 이 쿼리 한 곳만 고치면 다음 호출
    부터 바로 반영됨(2026-10-10, 외부 리뷰가 "BM25 캐시 무효화"를 걱정했는데
    실제로는 캐시 자체가 없어서 해당 없음).

    이 함수의 `course_filter`는 다른 조회 함수들과 달리 원래 `WHERE ...`
    형태였다(이 쿼리엔 기본 WHERE가 없어서) -- `_ACTIVE_SESSION_FILTER`가
    `AND ...`로 시작하는 공통 조각이라 그대로 이어붙이면 `course_id`가 없을
    때 `WHERE AND NOT EXISTS(...)` 같은 구문 오류가 난다(외부 리뷰 지적,
    실제로 맞는 함정이었음). 그래서 이 함수만 기본 `WHERE`를
    `_ACTIVE_SESSION_FILTER`가 쓰도록 바꾸고, `course_id`는 `AND`로 뒤에 붙인다.

    `content != ''` 조건(2026-10-10 추가)은 원래 `WHERE true`였던 자리 --
    `record_line_deletion()`이 "삭제됨" 표시로 남기는 빈 내용 행을 BM25
    코퍼스에서 제외한다(벡터 쪽은 그 행의 embedding이 NULL이라 이미
    `search_similar_chunks`에서 자동으로 빠짐, BM25는 embedding을 안 봐서
    따로 걸러줘야 함).
    """
    pool = get_pool()
    course_and = "AND course_id = %(course_id)s" if course_id else ""
    query = f"""
        SELECT content, source, origin_ref
        FROM note_chunks
        WHERE content != ''
        {_ACTIVE_SESSION_FILTER}
        {course_and}
        ORDER BY id;
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, {"course_id": course_id})
            columns = [c.name for c in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]


def is_file_already_ingested(path: str) -> bool:
    """이 파일에서 나온 청크가 note_chunks에 이미 있는지 확인.

    origin_ref는 항상 "{path}:..." 형태로 저장되니(어댑터마다 :page1, :chunk0 등
    접미사만 다름), 그 접두사로 매칭한다. `ingest_folder.py`가 폴더를 반복
    스캔할 때 이미 처리된 파일을 중복으로 또 넣지 않도록 하는 용도.

    경로를 "data/<source>/..." 형태로 정규화해서 비교한다 -- 구분자(\\\\ vs /)뿐
    아니라 절대경로/상대경로 차이도 같은 문제를 일으킨다는 게 실제로 확인됐다
    (2026-10-01: 예전에 절대경로로 수동 인제스트했던 49~57회차 파일이, 이번에
    상대경로로 비교하는 ingest_folder.py 때문에 "처리 안 된 파일"로 오판돼
    9개 파일, 280개 청크가 통째로 중복 적재됨 -- CLAUDE.md 참고). data/ 하위
    경로만 기준으로 비교하면 호출 시점의 작업 디렉터리나 절대/상대 표기 방식에
    관계없이 같은 파일을 같은 파일로 인식한다.
    """
    pool = get_pool()
    normalized = normalize_data_path(path) + ":%"
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                r"""
                SELECT 1 FROM note_chunks
                WHERE regexp_replace(replace(origin_ref, '\', '/'), '^.*(data/(personal|base|stt_reference)/)', '\1')
                      LIKE %(prefix)s
                LIMIT 1;
                """,
                {"prefix": normalized},
            )
            return cur.fetchone() is not None


def normalize_data_path(path: str) -> str:
    """경로를 "data/<personal|base|stt_reference>/..." 형태로 정규화.

    절대경로든 상대경로든, 구분자가 \\\\든 /든 상관없이 "data/" 이후 부분만
    남긴다. is_file_already_ingested()의 DB 쪽 정규화(SQL regexp_replace)와
    반드시 같은 규칙을 써야 한다. `adapters.py`도 origin_ref를 만들 때 이 함수로
    먼저 정규화한 경로를 쓴다 -- 체크 시점뿐 아니라 **생성 시점부터** 일관된
    형태여야 절대/상대경로가 섞여 들어와도 같은 파일이 같은 origin_ref로
    남는다(2026-10-02, 중복 적재 사고 재발 방지 보강 -- CLAUDE.md 참고).
    """
    forward = path.replace("\\", "/")
    match = re.search(r"data/(personal|base|stt_reference)/.*", forward)
    return match.group(0) if match else forward


def get_today_hints(course_id: str | None) -> list[dict]:
    """오늘(서버 기준 날짜) Tier2가 개입해 힌트를 만든 필기들을 오래된순으로 반환.

    프론트엔드의 "오늘의 요약" 패널용 -- `note_chunks`에 세션 단위 컬럼이
    따로 없어서(세션은 인메모리 작업기억 범위라 영구 저장 안 함, CLAUDE.md
    6번 섹션 참고) 대신 실제 처리 시각(`created_at`)의 날짜로 "오늘"을
    판단한다. `routing_decision='tier2'`인 것만(= 실제로 힌트가 만들어진
    것만) 가져온다 -- Tier1만 거친 조용한 필기는 여기 안 뜬다.

    오래된순(ASC)으로 반환 -- 왼쪽 필기창과 같은 시간축(위→아래로 갈수록
    최신)을 맞추기 위함(2026-10-02 실측 버그: DESC로 뒀다가 최신 힌트가
    위에 쌓여 필기 흐름과 반대 방향이 되는 문제 발견, CLAUDE.md 참고).

    `created_at`을 KST로 변환해서 "오늘"을 판단한다(2026-10-10, 외부 리뷰
    지적 -- `SHOW timezone`으로 직접 확인해보니 DB 세션 타임존이 UTC라,
    `created_at::date = CURRENT_DATE`를 그대로 쓰면 자정~오전9시(KST) 사이에
    쓴 필기가 "오늘"이 아니라 "어제"로 빠지는 실제 버그였음).
    """
    pool = get_pool()
    course_filter = "AND course_id = %(course_id)s" if course_id else ""
    query = f"""
        SELECT content, tier2_result, tier2_sources, created_at
        FROM note_chunks
        WHERE source = 'personal'
          AND routing_decision = 'tier2'
          AND tier2_result IS NOT NULL
          AND (created_at AT TIME ZONE 'Asia/Seoul')::date = (now() AT TIME ZONE 'Asia/Seoul')::date
          {course_filter}
          {_ACTIVE_SESSION_FILTER}
        ORDER BY created_at ASC;
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, {"course_id": course_id})
            columns = [c.name for c in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]


def get_all_notes(course_id: str | None) -> list[dict]:
    """이 과목의 personal 필기 전체(날짜 제한 없이)를 오래된순으로 반환.

    2026-10-08 전까지는 `created_at::date = CURRENT_DATE`로 "오늘"만 걸러서
    새로고침 시 복원했음(구 이름 get_today_notes) -- 그런데 서비스를 껐다가
    다음날 다시 켜면 화면이 다시 텅 비어서, "이어서 쓰는 필기장"이라는
    서비스 취지와 맞지 않는다는 지적 반영(사용자: "종료하고 다시 서비스를
    시작할 땐 기존 필기 기록들이 떠있는 상태여야 맞다"). 날짜 필터를 완전히
    제거해서 과목의 전체 히스토리를 복원한다 -- get_today_hints()와 달리
    tier2로 간 것만 거르지 않고 personal 소스 전체를 가져온다(2026-10-02,
    새로고침 시 가짜 데모 문단이 아니라 실제 필기가 보여야 한다는 지적 반영).

    `origin_ref`가 `ask:`로 시작하는 것(필기창이 아니라 "지금 바로 물어보기"
    입력창에서 온 즉석 질문)은 제외한다 -- 안 그러면 새로고침할 때 질문이
    필기창에 필기 한 줄처럼 복원돼버림(2026-10-02 실측 버그, api.py 참고).

    note_chunks는 RAG 재료용으로 append-only가 원칙이라(수정해도 새 행이
    또 쌓임), 같은 필기 줄을 고쳐서 여러 번 제출하면 그 모든 버전이 그대로
    이 쿼리에 걸려서 새로고침할 때마다 같은 내용이 여러 줄로 중복 표시되는
    버그가 있었다(2026-10-02 실측). `client_line_id`가 있는 행은 그 줄별로
    가장 최근 버전(rn=1)만 남기고, 없는 행(업로드/재생 스크립트처럼 한 번만
    제출되는 경우)은 전부 그대로 둔다.

    `origin_ref`도 반환한다 -- 프론트엔드가 여기서 "작성 단위"(예:
    `simulate_day.py`가 넣은 `sim:YYYYMMDD:...`의 날짜)를 뽑아내서 필기
    타임라인에 구분선을 그리는 데 쓴다(2026-10-02, "날짜/단원처럼 작성
    단위가 바뀌는 지점을 구분해달라"는 요청 반영). 실시간 입력(`live:...`)은
    이 쿼리가 파싱 가능한 날짜가 없는 origin_ref라 프론트엔드가 그냥 구분선
    없이 둔다.

    `session_id`도 반환한다(2026-10-08) -- 프론트엔드가 같은 날 안에서
    "새 필기 시작"으로 나눈 세션 경계를 재구성해서, 새로고침해도 그 구분선이
    사라지지 않고 "N번째 필기"로 다시 그려지도록 하는 데 쓴다. 이전엔
    session_id가 DB에만 남고(중복 힌트 방지용) 화면 재구성에는 전혀
    안 쓰여서, 새로고침하면 같은 날 여러 세션이 다시 하나로 뭉쳐 보이는
    문제가 있었음.

    `deleted_sessions`에 올라간 session_id는 제외한다(2026-10-09) -- "세션
    삭제" 버튼은 더 이상 note_chunks에서 실제로 지우지 않는다(사용자 결정:
    "그것도 데이터인데 아예 복구 못 하고 기록도 못 보게 없애는 건 아닌 것
    같다" -- 화면/타임라인에서는 사라지고 "N번째 필기세션 삭제됨" 로그만
    남지만, 원본 행은 DB에 그대로 남아있음). 그래서 화면 재구성(이 함수)
    쪽에서 걸러줘야 "삭제됐다"는 게 실제로 반영된다 -- 안 그러면 새로고침할
    때 지운 세션이 다시 나타나 버림.

    `content != ''`(2026-10-10 추가): 같은 `client_line_id`의 최신 버전이
    빈 내용이면 그 줄 전체를 화면에서 뺀다 -- `record_line_deletion()`이
    과거 줄을 지우고 Enter 누르면 남기는 "삭제됨" 표시. RAG 검색 쪽
    (search_similar_chunks 등)은 이 필터와 무관하게 이전 버전(실제 내용)을
    계속 검색 대상으로 쓴다 -- 화면 삭제가 RAG 코퍼스에는 전파되지 않는
    의도된 설계(사용자 방침: "RAG는 지식 로그라 쌓여있는 게 좋다").
    """
    pool = get_pool()
    course_filter = "AND course_id = %(course_id)s" if course_id else ""
    query = f"""
        WITH all_notes AS (
            SELECT content, tier2_result, tier2_sources, created_at, client_line_id, origin_ref, session_id,
                   inserted_out_of_place,
                   ROW_NUMBER() OVER (
                       PARTITION BY client_line_id ORDER BY created_at DESC
                   ) AS rn
            FROM note_chunks
            WHERE source = 'personal'
              AND origin_ref NOT LIKE 'ask:%%'
              {course_filter}
              {_ACTIVE_SESSION_FILTER}
        )
        SELECT content, tier2_result, tier2_sources, created_at, client_line_id, origin_ref, session_id,
               inserted_out_of_place
        FROM all_notes
        WHERE (client_line_id IS NULL OR rn = 1) AND content != ''
        ORDER BY created_at ASC;
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, {"course_id": course_id})
            columns = [c.name for c in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]


def get_session_time_range(session_id: str) -> datetime | None:
    """세션 삭제 버튼용(2026-10-09, 설계 변경) -- 이 session_id에 속한
    personal 필기의 가장 이른 시각을 반환. 한 건도 없으면 None.

    2026-10-08엔 이 자리에서 실제로 `DELETE FROM note_chunks`를 했었는데,
    사용자 결정으로 물리 삭제를 그만뒀다 -- "그것도 데이터인데 아예 복구
    못 하고 기록도 못 보게 없애는 건 아닌 것 같다. 그 경우의 수도 많지
    않을 거고(보통 하루에 과목당 세션 하나)". note_chunks는 이제 손대지
    않고, 화면/검색에서 숨기는 역할은 `deleted_sessions` 테이블 하나가
    전부 담당한다(`get_all_notes()`의 제외 조건 참고). 이 함수는 그 묘비에
    적을 "원래 시작 시각"만 조회한다.
    """
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT MIN(created_at) FROM note_chunks "
                "WHERE session_id = %(session_id)s AND source = 'personal';",
                {"session_id": session_id},
            )
            return cur.fetchone()[0]


def insert_deleted_session(
    session_id: str, course_id: str | None, day_key: str, ordinal: int, session_started_at: str
) -> None:
    """세션 삭제 "묘비"(tombstone) 기록(2026-10-08, 2026-10-09에 호출부 변경
    -- 이제 note_chunks를 지우지 않으므로 `get_session_time_range()` 조회
    직후 호출). 원래 문제: 세션을 지우면(과거엔 실제 DELETE) 그 세션의
    note_chunks 행 자체가 사라져서 "그날 몇 번째 세션이었는지"를 알 길이
    없어짐(사용자 지적: "동일 날짜에 세션이 서너 번 생겼는데 중간 걸 지우면
    번호가 꼬일 수 있다 -- 생성 시점 번호는 고정하고, 삭제된 건 '몇 번째가
    삭제됨'이라고 타임라인에 남기면 재넘버링이 필요 없다"). 지금은 행이
    실제로 안 지워지지만, 화면에서 숨기는 기준 자체가 이 테이블이라 여전히
    필요하다. `ordinal`은 생성 시점에 매겨진 번호를 그대로 받아서 저장
    (재계산 안 함), `day_key`/`session_started_at`은 호출부(server.py)가
    `get_session_time_range()` 결과를 KST로 변환해서 넘겨준다.
    """
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO deleted_sessions "
                "(session_id, course_id, day_key, ordinal, session_started_at) "
                "VALUES (%(session_id)s, %(course_id)s, %(day_key)s, %(ordinal)s, %(session_started_at)s) "
                "ON CONFLICT (session_id) DO NOTHING;",
                {
                    "session_id": session_id,
                    "course_id": course_id,
                    "day_key": day_key,
                    "ordinal": ordinal,
                    "session_started_at": session_started_at,
                },
            )


def get_deleted_sessions(course_id: str | None) -> list[dict]:
    """이 과목에서 삭제된 세션 묘비 전체를 반환 -- loadAllNotes()가 필기
    타임라인에 "N번째 필기세션 삭제됨" 구분선을 끼워 넣는 데 쓴다.
    """
    pool = get_pool()
    course_filter = "AND course_id = %(course_id)s" if course_id else ""
    query = f"""
        SELECT day_key, ordinal, session_started_at
        FROM deleted_sessions
        WHERE true {course_filter}
        ORDER BY session_started_at ASC;
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, {"course_id": course_id})
            columns = [c.name for c in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]


def log_session_event(session_id: str, event_type: str, note_count: int | None = None) -> None:
    """세션 작업기억의 생애주기를 타임스탬프와 함께 기록.

    화면에 명시적으로 드러나는 기능은 아니지만(사용자에게 안 보임), 나중에
    "세션이 보통 얼마나 지속되는지, 몇 개나 쌓이는지" 분석하거나 유휴 타임아웃
    자동 종료 기능을 설계할 때 필요한 데이터라 지금부터 남겨둔다.
    """
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO session_events (session_id, event_type, note_count) "
                "VALUES (%(session_id)s, %(event_type)s, %(note_count)s);",
                {"session_id": session_id, "event_type": event_type, "note_count": note_count},
            )


def insert_note_chunk(row: dict) -> None:
    """note_chunks에 판단 결과 한 행을 저장.

    row는 테이블 컬럼명과 동일한 키를 가진 dict를 기대한다 (nodes.persist_note가
    State에서 컬럼명 기준으로 매핑해서 넘겨줌). Tier1만 거쳤든 Tier2까지 갔든
    공통으로 호출된다 -- 판단 로그(top_score, routing_decision 등)는 항상 남아야 함.

    `origin_ref`에 UNIQUE 제약이 걸려있어(2026-10-02 추가), 같은 origin_ref로
    또 삽입을 시도하면 `ON CONFLICT DO NOTHING`으로 조용히 무시된다 -- 중복
    적재 버그가 또 발생하더라도 DB 단에서 최후 방어선 역할을 함(스크립트 재실행
    실수 등). 단, 이 안전장치가 제대로 작동하려면 origin_ref 자체가 생성
    시점부터 정규화돼 있어야 하므로(`normalize_data_path` 참고), 호출 전에
    이미 정규화된 origin_ref를 넘기는 게 전제다.
    """
    pool = get_pool()
    query = """
        INSERT INTO note_chunks (
            content, source, medium, course_id, note_timestamp, origin_ref,
            top_score, is_question, is_abbreviated, is_new_concept,
            session_duplicate_score, routing_decision, tier2_result, embedding,
            client_line_id, hint_category, top_match_feedback_value,
            top_match_feedback_days_ago, session_id, tier2_sources, inserted_out_of_place
        ) VALUES (
            %(content)s, %(source)s, %(medium)s, %(course_id)s,
            %(note_timestamp)s, %(origin_ref)s,
            %(top_score)s, %(is_question)s, %(is_abbreviated)s, %(is_new_concept)s,
            %(session_duplicate_score)s, %(routing_decision)s, %(tier2_result)s, %(embedding)s::vector,
            %(client_line_id)s, %(hint_category)s, %(top_match_feedback_value)s,
            %(top_match_feedback_days_ago)s, %(session_id)s, %(tier2_sources)s, %(inserted_out_of_place)s
        )
        ON CONFLICT (origin_ref) DO NOTHING;
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, row)
            if cur.rowcount == 0:
                print(f"[insert_note_chunk] 중복 origin_ref라 건너뜀: {row.get('origin_ref')!r}")


def record_line_deletion(
    client_line_id: str, course_id: str | None, session_id: str | None, origin_ref: str
) -> None:
    """필기 줄 하나를 "삭제됨"으로 기록 -- 빈 내용으로 새 버전을 추가한다.

    2026-10-10, 사용자 요청: "수정이 가능한데 삭제가 안 되는 건 이상하다 --
    최소한 과거 줄 지우고 엔터를 누르면 그것도 수정과 마찬가지로 반영돼야
    한다." 지금까지 `submitNote()`는 빈 내용이면 그냥 조용히 무시했어서
    (empty text는 서버에 전혀 전달 안 됨), 화면에서 지워도 새로고침하면
    원래 내용이 그대로 복원되는 문제가 있었음.

    `insert_note_chunk()`(전체 LangGraph 파이프라인 거친 뒤 호출됨)를 안
    쓰고 이 함수로 바로 가는 이유: 빈 내용을 임베딩하거나 LLM 판단(질문/축약
    여부)에 태우는 건 무의미하고 비용 낭비. `embedding`은 일부러 NULL로
    남긴다 -- `search_top_similarity`/`search_similar_chunks`가 이미
    `WHERE embedding IS NOT NULL`로 걸러서 자동으로 검색 대상에서 빠지고,
    `fetch_corpus_for_bm25`는 `content != ''` 필터로 따로 뺀다.

    이전 버전(실제 내용이 있던 행)은 전혀 안 건드린다 -- 사용자 방침:
    "RAG에 되먹임되는 데이터는 지식 로그라서, 추가/수정/삭제 전부 쌓여있는
    게 개인화에 좋다. 화면(일반 조회)에서는 삭제/수정이 반영되지만, RAG
    검색 자료는 그대로 둔다"는 의도적 설계 -- `get_all_notes()`가 같은
    `client_line_id`의 최신 버전(이 빈 행)을 골라서 화면에서 빼주는 역할만
    하고, RAG 검색 쿼리들은 이 함수가 추가하는 행과 무관하게 이전 버전들을
    계속 검색 대상으로 쓴다(append-only 원칙 유지).
    """
    pool = get_pool()
    query = """
        INSERT INTO note_chunks (content, source, medium, course_id, note_timestamp, origin_ref, client_line_id, session_id)
        VALUES ('', 'personal', 'txt', %(course_id)s, now(), %(origin_ref)s, %(client_line_id)s, %(session_id)s)
        ON CONFLICT (origin_ref) DO NOTHING;
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                query,
                {
                    "course_id": course_id,
                    "origin_ref": origin_ref,
                    "client_line_id": client_line_id,
                    "session_id": session_id,
                },
            )


def log_failed_ingestion(origin_ref: str, error_type: str, error_message: str) -> None:
    """인제스트 실패를 조용히 흘려보내지 않고 `failed_ingestions`에 기록.

    비전 LLM 페이지 OCR처럼 개별 페이지/청크 단위로 실패할 수 있는 경로에서
    쓴다 -- 지금까지는 콘솔 print()로만 남아서 로그 파일을 뒤져야 뭐가
    실패했는지 알 수 있었는데, 테이블에 남기면 `SELECT * FROM failed_ingestions`
    한 줄로 전체 실패 목록을 보고 나중에 수동 보정할 수 있다(2026-10-02,
    "Dead Letter" 패턴 -- CLAUDE.md 참고).
    """
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO failed_ingestions (origin_ref, error_type, error_message) "
                "VALUES (%(origin_ref)s, %(error_type)s, %(error_message)s);",
                {"origin_ref": origin_ref, "error_type": error_type, "error_message": error_message},
            )


def get_sim_day_notes(course_id: str | None, period_key: str) -> list[dict]:
    """simulate_day.py가 `sim:{period_key}:...` origin_ref로 넣은 그 날 필기 전체(원문+힌트)를 시간순으로 반환.

    요약 생성 재료용 -- period_summaries가 캐시 미스일 때만 호출된다.
    지금은 origin_ref의 "sim:YYYYMMDD:" 접두사로만 날짜를 구분할 수 있어서
    (실시간 입력은 애초에 "오늘" 하루 범위를 안 벗어나 날짜 구분 자체가
    필요 없음, notemate_mockup.html의 unit divider와 동일한 전제) 이 함수도
    같은 가정을 쓴다.
    """
    pool = get_pool()
    course_filter = "AND course_id = %(course_id)s" if course_id else ""
    query = f"""
        SELECT content, tier2_result
        FROM note_chunks
        WHERE source = 'personal'
          AND origin_ref LIKE %(prefix)s
          {course_filter}
        ORDER BY created_at ASC;
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, {"prefix": f"sim:{period_key}:%", "course_id": course_id})
            columns = [c.name for c in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]


def get_period_summary(course_id: str | None, period_type: str, period_key: str) -> dict | None:
    """캐시된 기간 요약이 있으면 반환, 없으면 None -- 재생성 방지용."""
    pool = get_pool()
    pool_query = """
        SELECT summary, note_count, tier2_count, created_at
        FROM period_summaries
        WHERE period_type = %(period_type)s AND period_key = %(period_key)s
          AND course_id IS NOT DISTINCT FROM %(course_id)s;
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(pool_query, {"period_type": period_type, "period_key": period_key, "course_id": course_id})
            row = cur.fetchone()
            if row is None:
                return None
            columns = [c.name for c in cur.description]
            return dict(zip(columns, row))


def save_period_summary(
    course_id: str | None, period_type: str, period_key: str, summary: str, note_count: int, tier2_count: int
) -> None:
    """기간 요약 저장 -- (course_id, period_type, period_key) 조합당 한 번만 생성되게 UNIQUE 제약.

    period_type을 'day' 하나로 고정하지 않고 일반화해둔 이유: 나중에
    'week'/'unit' 등 다른 기간 단위 요약도 같은 테이블에 쌓아서, day 요약을
    원재료 삼아 한 단계 더 요약하는 계층 구조(2026-10-02 요청, "층위기억을
    위한 raw 데이터로도 쓸 수 있게")로 자연스럽게 확장 가능하게 하려는 것.
    """
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO period_summaries (course_id, period_type, period_key, summary, note_count, tier2_count)
                VALUES (%(course_id)s, %(period_type)s, %(period_key)s, %(summary)s, %(note_count)s, %(tier2_count)s)
                ON CONFLICT (course_id, period_type, period_key) DO NOTHING;
                """,
                {
                    "course_id": course_id,
                    "period_type": period_type,
                    "period_key": period_key,
                    "summary": summary,
                    "note_count": note_count,
                    "tier2_count": tier2_count,
                },
            )

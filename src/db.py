"""Supabase(Postgres) 연결 및 note_chunks 쿼리 헬퍼.

supabase-py(REST API) 대신 psycopg로 직접 연결한다 (CLAUDE.md 2026-09-29
결정 참고 -- 복합 SQL 쿼리 자유도 확보 목적). 커넥션 풀은 프로세스당 하나만
만들어 재사용한다.
"""

import os
import re

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


def search_top_similarity(embedding: list[float], course_id: str | None) -> float | None:
    """note_chunks에서 embedding과 가장 유사한 기존 행의 코사인 유사도(0~1에 가까움)를 반환.

    비교할 기존 행이 하나도 없으면(테이블이 비어있는 등) None을 반환한다 --
    "지원 자료가 없다"는 의미이므로 호출부(assess_note)에서 확신도 낮음으로 처리한다.

    파이썬 list를 psycopg에 그냥 넘기면 double precision[]로 직렬화돼서
    pgvector의 <=> 연산자가 못 알아본다 -- 그래서 %s::vector로 명시적으로
    캐스팅한다 (register_vector 어댑터 자동 등록보다 이 방식이 더 안정적).
    """
    pool = get_pool()
    course_filter = "AND course_id = %(course_id)s" if course_id else ""
    query = f"""
        SELECT 1 - (embedding <=> %(embedding)s::vector) AS similarity
        FROM note_chunks
        WHERE embedding IS NOT NULL
        {course_filter}
        ORDER BY embedding <=> %(embedding)s::vector
        LIMIT 1;
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, {"embedding": embedding, "course_id": course_id})
            row = cur.fetchone()
            return row[0] if row else None


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
    코퍼스를 불러와서 즉석으로 인덱스를 만든다(src/retrieval.py).
    """
    pool = get_pool()
    course_filter = "WHERE course_id = %(course_id)s" if course_id else ""
    query = f"""
        SELECT content, source, origin_ref
        FROM note_chunks
        {course_filter}
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
    """
    pool = get_pool()
    course_filter = "AND course_id = %(course_id)s" if course_id else ""
    query = f"""
        SELECT content, tier2_result, created_at
        FROM note_chunks
        WHERE source = 'personal'
          AND routing_decision = 'tier2'
          AND tier2_result IS NOT NULL
          AND created_at::date = CURRENT_DATE
          {course_filter}
        ORDER BY created_at ASC;
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, {"course_id": course_id})
            columns = [c.name for c in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]


def get_today_notes(course_id: str | None) -> list[dict]:
    """오늘(서버 기준 날짜) 실시간으로 입력된 필기 전체를 오래된순으로 반환.

    프론트엔드가 새로고침될 때 "빈 화면"이 아니라 실제로 오늘 쓴 내용을
    그대로 복원하기 위한 용도 -- get_today_hints()와 달리 tier2로 간 것만
    거르지 않고 personal 소스 전체를 가져온다(2026-10-02, 새로고침 시
    가짜 데모 문단이 아니라 실제 필기가 보여야 한다는 지적 반영).

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
    애초에 "오늘" 범위 밖으로 안 나가니 구분선이 필요 없고, 이 쿼리가
    파싱 가능한 날짜가 없는 origin_ref는 프론트엔드가 그냥 구분선 없이 둔다.
    """
    pool = get_pool()
    course_filter = "AND course_id = %(course_id)s" if course_id else ""
    query = f"""
        WITH todays AS (
            SELECT content, tier2_result, created_at, client_line_id, origin_ref,
                   ROW_NUMBER() OVER (
                       PARTITION BY client_line_id ORDER BY created_at DESC
                   ) AS rn
            FROM note_chunks
            WHERE source = 'personal'
              AND created_at::date = CURRENT_DATE
              AND origin_ref NOT LIKE 'ask:%%'
              {course_filter}
        )
        SELECT content, tier2_result, created_at, client_line_id, origin_ref
        FROM todays
        WHERE client_line_id IS NULL OR rn = 1
        ORDER BY created_at ASC;
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
            client_line_id
        ) VALUES (
            %(content)s, %(source)s, %(medium)s, %(course_id)s,
            %(note_timestamp)s, %(origin_ref)s,
            %(top_score)s, %(is_question)s, %(is_abbreviated)s, %(is_new_concept)s,
            %(session_duplicate_score)s, %(routing_decision)s, %(tier2_result)s, %(embedding)s::vector,
            %(client_line_id)s
        )
        ON CONFLICT (origin_ref) DO NOTHING;
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, row)
            if cur.rowcount == 0:
                print(f"[insert_note_chunk] 중복 origin_ref라 건너뜀: {row.get('origin_ref')!r}")


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

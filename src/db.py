"""Supabase(Postgres) 연결 및 note_chunks 쿼리 헬퍼.

supabase-py(REST API) 대신 psycopg로 직접 연결한다 (CLAUDE.md 2026-09-29
결정 참고 -- 복합 SQL 쿼리 자유도 확보 목적). 커넥션 풀은 프로세스당 하나만
만들어 재사용한다.
"""

import os

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
    normalized = _normalize_data_path(path) + ":%"
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


def _normalize_data_path(path: str) -> str:
    """경로를 "data/<personal|base|stt_reference>/..." 형태로 정규화.

    절대경로든 상대경로든, 구분자가 \\\\든 /든 상관없이 "data/" 이후 부분만
    남긴다. is_file_already_ingested()의 DB 쪽 정규화(SQL regexp_replace)와
    반드시 같은 규칙을 써야 하므로 별도 함수로 분리해 재사용한다.
    """
    import re

    forward = path.replace("\\", "/")
    match = re.search(r"data/(personal|base|stt_reference)/.*", forward)
    return match.group(0) if match else forward


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
    """
    pool = get_pool()
    query = """
        INSERT INTO note_chunks (
            content, source, medium, course_id, note_timestamp, origin_ref,
            top_score, is_question, is_abbreviated, is_new_concept,
            session_duplicate_score, routing_decision, tier2_result, embedding
        ) VALUES (
            %(content)s, %(source)s, %(medium)s, %(course_id)s,
            %(note_timestamp)s, %(origin_ref)s,
            %(top_score)s, %(is_question)s, %(is_abbreviated)s, %(is_new_concept)s,
            %(session_duplicate_score)s, %(routing_decision)s, %(tier2_result)s, %(embedding)s::vector
        );
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, row)

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


def _normalize_course_ids(course_id: str | list[str] | None) -> list[str] | None:
    """노트 묶음(RAG 공유그룹, 2026-10-17 야간) 지원용 정규화 헬퍼.

    기존 호출부는 전부 단일 문자열(`course_id="sesac_bootcamp"`)을 넘기고,
    묶음 검색이 필요한 새 호출부는 리스트(`["a", "b", "c"]`)를 넘긴다 --
    둘 다 받아서 항상 리스트(또는 필터 없음 의미의 None)로 통일한다.
    """
    if not course_id:
        return None
    return [course_id] if isinstance(course_id, str) else list(course_id)


def search_top_similarity(embedding: list[float], course_id: str | list[str] | None) -> dict | None:
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

    `course_id`에 리스트를 넘기면(노트 묶음, 2026-10-17 야간) 그 묶음 전체에서
    가장 유사한 걸 찾는다 -- "관련 노트끼리는 서로 친숙도 판단에도 참고한다"는
    설계 의도 그대로(새 글을 쓸 때 묶인 다른 노트에 비슷한 내용이 있었으면
    그것도 "이미 아는 내용"으로 쳐준다).
    """
    pool = get_pool()
    course_ids = _normalize_course_ids(course_id)
    course_filter = "AND course_id = ANY(%(course_ids)s)" if course_ids else ""
    query = f"""
        SELECT 1 - (embedding <=> %(embedding)s::vector) AS similarity,
               feedback_value, feedback_at, hint_excluded
        FROM note_chunks
        WHERE embedding IS NOT NULL
        {course_filter}
        {_ACTIVE_SESSION_FILTER}
        ORDER BY embedding <=> %(embedding)s::vector
        LIMIT 1;
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, {"embedding": embedding, "course_ids": course_ids})
            row = cur.fetchone()
            if not row:
                return None
            return {
                "similarity": row[0],
                "feedback_value": row[1],
                "feedback_at": row[2],
                "hint_excluded": row[3],
            }


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


def record_hint_exclusion(origin_ref: str, excluded: bool) -> bool:
    """"힌트 예외처리"(2026-10-17 야간 설계, deferred-hint-feedback-plus-minus-logic.md 참고) --

    이 힌트가 달린 원본 필기 행이 양식/템플릿이라 앞으로 힌트 생성 대상이
    아니라고 표시(또는 해제)한다. `feedback_value`(+/-/✓)와는 **별개 컬럼**으로
    둔 이유: 의미가 다른 신호라(품질 반응 vs 범주 제외) 같은 컬럼을 공유하면
    서로 덮어써서 공존할 수 없게 됨. 체크 해제 시 `excluded=False`로 다시
    호출하면 되돌릴 수 있다(원복 가능, 데이터 손실 없음).
    """
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE note_chunks SET hint_excluded = %(excluded)s, "
                "hint_excluded_at = CASE WHEN %(excluded)s THEN now() ELSE NULL END "
                "WHERE origin_ref = %(origin_ref)s;",
                {"excluded": excluded, "origin_ref": origin_ref},
            )
            conn.commit()
            return cur.rowcount > 0


def get_hint_excluded_origin_refs() -> list[str]:
    """현재 `hint_excluded=TRUE`로 표시된 origin_ref 전체를 반환.

    로그 파일이 아니라 DB 컬럼이 곧 정본(canonical state)이라, 체크상태
    복원용으로 이 목록 하나면 충분하다(hint_feedback_state처럼 로그를
    순서대로 훑어 재구성할 필요 없음).
    """
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT origin_ref FROM note_chunks WHERE hint_excluded = TRUE;")
            return [row[0] for row in cur.fetchall()]


def search_similar_chunks(
    embedding: list[float], course_id: str | list[str] | None, k: int = 5, exclude_questions: bool = True
) -> list[dict]:
    """note_chunks에서 embedding과 유사한 상위 k개 행을 반환 (content/source/origin_ref 포함).

    assess_note의 search_top_similarity()는 점수 하나만 필요해서 top-1만 보지만,
    Tier2 딥 에이전트는 실제 근거로 LLM에 넘길 내용 자체가 필요해서 여러 개를 가져온다.

    `is_question IS NOT TRUE`(2026-10-15, 외부 리뷰 Q4 반영): 질문으로 저장된
    과거 필기 행은 근거로 쓰지 않는다 -- "요즘 환율이 어떻게 돼?"와 "요즘
    비트코인 시세가 어떻게 돼?"처럼 문장 틀만 같은 서로 다른 질문이 서로의
    근거가 돼버리는 자기참조 오탐을 구조적으로 막는다. `= FALSE`가 아니라
    `IS NOT TRUE`를 쓰는 이유: 업로드 자료(source='base'/'stt_reference')는
    판단 자체를 안 거쳐서(embed_and_store 경로) is_question이 항상 NULL인데,
    `= FALSE`로 쓰면 NULL 비교가 전부 거짓이 되어 강의자료가 통째로 걸러져
    버린다(외부 리뷰가 지적한 함정, 실제로 맞는 내용). search_top_similarity()
    (라우팅 확신도 계산용)에는 이 필터를 일부러 적용하지 않는다 -- 적용하면
    라우팅 자체가 바뀌어서 3차 대비 4차 비교에 "근거 필터링"과 "라우팅 로직
    변경"이 뒤섞이게 된다(힌트 출처 표시 전용 경로에만 영향을 한정).

    `exclude_questions=False`(2026-10-17 추가, 힌트 피드백 반영 로직 버그 수정)로
    호출하면 위 필터를 끈다 -- 실사용 중 발견: "바로 물어보기"/명시적 질문
    줄은 전부 is_question=True라서, 그 줄에 준 +/- 피드백이 다음에 비슷한
    질문이 들어와도 이 필터에 걸려 영원히 검색이 안 되는 구조적 버그였음(사용자가
    같은 질문에 `-`를 두 번 줘도 전혀 반영 안 되는 걸 실측으로 재현). "근거
    표시용 검색"(위 문단 목적)과 "피드백 이력 확인용 검색"은 서로 다른
    목적이라 둘을 분리 -- 피드백 확인은 질문이든 아니든 과거에 비슷한 내용에
    준 피드백을 찾아야 하므로 이 필터가 없어야 한다.
    """
    # feedback_value/tier2_result 추가(2026-10-17, 힌트 피드백 반영 로직) --
    # 과거에 이 내용과 비슷한 힌트가 +/-를 받았는지 tier2_deep_agent가 알아야
    # "다르게 다시 시도"/"이 방식 유지" 지시를 프롬프트에 넣을 수 있다.
    # 2026-10-17 야간: 노트 묶음(RAG 공유그룹) -- course_id에 리스트를 넘기면
    # 묶인 노트 전체를 Tier2 근거 검색 범위로 삼는다.
    pool = get_pool()
    course_ids = _normalize_course_ids(course_id)
    course_filter = "AND course_id = ANY(%(course_ids)s)" if course_ids else ""
    question_filter = "AND is_question IS NOT TRUE" if exclude_questions else ""
    query = f"""
        SELECT content, source, origin_ref, feedback_value, tier2_result,
               1 - (embedding <=> %(embedding)s::vector) AS similarity
        FROM note_chunks
        WHERE embedding IS NOT NULL
        {question_filter}
        {course_filter}
        {_ACTIVE_SESSION_FILTER}
        ORDER BY embedding <=> %(embedding)s::vector
        LIMIT %(k)s;
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, {"embedding": embedding, "course_ids": course_ids, "k": k})
            columns = [c.name for c in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]


def fetch_corpus_for_bm25(course_id: str | list[str] | None) -> list[dict]:
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

    `is_question IS NOT TRUE`(2026-10-15): search_similar_chunks()와 같은
    이유(자기참조 오탐 방지) -- BM25 코퍼스에도 질문 행이 남아있으면 벡터
    쪽만 걸러봐야 소용없다(_filter_relevant_docs가 벡터+BM25 교차 인정이라
    한쪽이라도 질문 행을 인정하면 뚫림). `= FALSE`가 아니라 `IS NOT TRUE`를
    쓰는 이유도 동일(NULL인 업로드 자료를 제외하면 안 됨).
    """
    # feedback_value/tier2_result 추가(2026-10-17) -- search_similar_chunks()와
    # 같은 이유(힌트 피드백 반영 로직이 BM25 매치 경로에서도 동작해야 함).
    # 2026-10-17 야간: 노트 묶음(RAG 공유그룹) -- course_id에 리스트를 넘기면
    # 묶인 노트 전체를 BM25 코퍼스 범위로 삼는다(search_similar_chunks와 짝).
    pool = get_pool()
    course_ids = _normalize_course_ids(course_id)
    course_and = "AND course_id = ANY(%(course_ids)s)" if course_ids else ""
    query = f"""
        SELECT content, source, origin_ref, feedback_value, tier2_result
        FROM note_chunks
        WHERE content != ''
          AND is_question IS NOT TRUE
        {_ACTIVE_SESSION_FILTER}
        {course_and}
        ORDER BY id;
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, {"course_ids": course_ids})
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


def get_recent_session_boundary(course_id: str | None, session_limit: int) -> datetime | None:
    """최근 N개 "활성 단위"(세션 또는 날짜) 중 가장 오래된 것이 속한 날의
    자정(KST) 시각을 반환한다 -- 서비스 시작 시 전체 히스토리를 한꺼번에
    불러오지 않고 최근 분량만 먼저 보여주기 위한 경계값(2026-10-14, 사용자
    지적: "서비스 시작할 때 로딩이 너무 오래 걸린다").

    단위(unit) = session_id가 있으면 그 값, 없으면(세션 개념이 생기기 전
    과거 데이터) 날짜(KST)로 대체 -- 프론트엔드 날짜 구분선이 사실상 그
    역할을 이미 하고 있어서 자연스러운 대체.

    삭제된 세션(deleted_sessions)은 집계에서 아예 제외한다 -- 테스트하다
    세션을 여러 번 만들었다 지운 경우까지 "최근 N개"에 들어가면 실제 보여줄
    분량이 의도보다 줄어든다(2026-10-14 사용자 지적: "삭제세션은 자르는
    기준에서 제외하고 분량을 계산해야겠지").

    경계는 "그 단위가 시작한 세션/날짜 단위"가 아니라 반드시 "그 날의
    자정"으로 반올림한다 -- 세션 단위로 정확히 끊으면, 하루에 세션이 여러
    번 있었을 때 그 날의 첫 세션이 잘려나가 "최근 창의 첫 세션"이 실제로는
    그 날의 2번째/3번째 세션인데도 화면엔 "새 필기 시작"(1번째)으로 잘못
    표시되는 문제가 생긴다. 항상 날짜 단위로 자르면 이 경계가 날짜 경계와도
    겹쳐서, 나중에 그 이전 분량을 배경에서 불러와 위에 끼워 넣을 때도 두
    구간의 날짜/세션 번호 매기기가 서로 전혀 간섭하지 않는다(완전히 독립된
    날짜들이라 번호를 재계산할 필요가 없음).

    전체 활성 단위 수가 session_limit 이하면 None을 반환한다 -- "자를 필요
    없이 전부 로딩하라"는 뜻으로 해석된다(히스토리가 작은 과목/신규 과목은
    이 분할 로딩 자체가 무의미하므로 자연히 기존 동작 그대로 유지됨).

    "바로 물어보기"(`ask:` origin_ref)도 2026-10-14부터 단위 집계에 포함한다
    -- get_all_notes()와 같은 이유(질문도 타임라인의 일부). 질문만 있고
    일반 필기가 없는 세션도 "활성 세션 1개"로 정상 집계된다.

    `sim:`(재생 테스트) 접두사 데이터는 2026-10-17부터 집계에서 제외한다 --
    6차 사이클 재생이 자정을 넘겨서까지 실시간으로 계속 돌면서, 재생이 만든
    "하루치 세션"들이 실제 삽입 시각 기준으로 "최근 N개"에 끼어들어 경계가
    오늘 자정으로 잡히고, 그 결과 거의 전체 재생 데이터(2,303줄 중 1,999줄)가
    "최근"으로 분류되는 문제가 실측으로 확인됨. 재생 테스트 데이터는 실제
    사람이 지금 막 쓴 세션이 아니므로 "최근성" 판단에서 원천적으로 빼는 게
    맞음(사용자 확인 후 반영).
    """
    pool = get_pool()
    course_filter = "AND course_id = %(course_id)s" if course_id else ""
    query = f"""
        WITH units AS (
            SELECT
                COALESCE(session_id::text, to_char(created_at AT TIME ZONE 'Asia/Seoul', 'YYYYMMDD')) AS unit_key,
                created_at
            FROM note_chunks
            WHERE source = 'personal'
              AND content != ''
              AND origin_ref NOT LIKE 'sim:%%'
              {course_filter}
              {_ACTIVE_SESSION_FILTER}
        ),
        unit_bounds AS (
            SELECT unit_key, MIN(created_at) AS started_at
            FROM units
            GROUP BY unit_key
        ),
        ranked AS (
            SELECT started_at, ROW_NUMBER() OVER (ORDER BY started_at DESC) AS rn
            FROM unit_bounds
        )
        SELECT (date_trunc('day', started_at AT TIME ZONE 'Asia/Seoul') AT TIME ZONE 'Asia/Seoul')
        FROM ranked
        WHERE rn = %(session_limit)s;
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, {"course_id": course_id, "session_limit": session_limit})
            row = cur.fetchone()
            return row[0] if row else None


def get_all_notes(
    course_id: str | None, after: datetime | None = None, before: datetime | None = None
) -> list[dict]:
    """이 과목의 personal 필기 전체(날짜 제한 없이)를 오래된순으로 반환.

    2026-10-08 전까지는 `created_at::date = CURRENT_DATE`로 "오늘"만 걸러서
    새로고침 시 복원했음(구 이름 get_today_notes) -- 그런데 서비스를 껐다가
    다음날 다시 켜면 화면이 다시 텅 비어서, "이어서 쓰는 필기장"이라는
    서비스 취지와 맞지 않는다는 지적 반영(사용자: "종료하고 다시 서비스를
    시작할 땐 기존 필기 기록들이 떠있는 상태여야 맞다"). 날짜 필터를 완전히
    제거해서 과목의 전체 히스토리를 복원한다 -- get_today_hints()와 달리
    tier2로 간 것만 거르지 않고 personal 소스 전체를 가져온다(2026-10-02,
    새로고침 시 가짜 데모 문단이 아니라 실제 필기가 보여야 한다는 지적 반영).

    `origin_ref`가 `ask:`로 시작하는 것("지금 바로 물어보기"로 들어온 즉석
    질문)도 2026-10-14부터 포함한다 -- 사용자 요구: "질문도 학습의 일종,
    입력 통로가 다를 뿐 타임라인 기록에 포함되는 게 맞다. 필기와 힌트표시
    통합, 작동 방식도 동일(마커 형태/색깔만 구분용으로 다름)". 2026-10-02엔
    반대로 제외해야 했는데(그때는 질문이 필기 한 줄처럼 그대로 복원돼버려
    실제 필기와 구분이 안 됐던 버그), 그 문제는 프론트엔드가 `origin_ref`의
    `ask:` 접두사를 보고 필기와 다른 모양(파란 로그 줄 + 다른 색 마커)으로
    렌더링하도록 바뀌면서 해결됐다(notemate_mockup.html의 `isAskRow()`/
    `renderAskLineFast()` 참고) -- 이제는 제외할 이유가 없어짐.

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

    `medium != 'paper_ocr'`(2026-10-17 추가): 종이 필기 스캔본(61페이지)은
    한 번에 일괄 백필된 참고자료라, "순차적으로 쌓인 필기 흐름"을 보여주는
    이 타임라인에 끼워 넣으면 어색함(사용자 지적: "그건 한번에 넣은
    거잖아"). RAG 검색 쪽(search_similar_chunks 등)은 이 필터와 무관하게
    계속 검색 대상으로 씀 -- 화면 표시만 빠지고 힌트 생성 근거로는 그대로
    쓰이는, `content != ''` 삭제 처리와 같은 성격의 분리.

    `content != ''`(2026-10-10 추가): 같은 `client_line_id`의 최신 버전이
    빈 내용이면 그 줄 전체를 화면에서 뺀다 -- `record_line_deletion()`이
    과거 줄을 지우고 Enter 누르면 남기는 "삭제됨" 표시. RAG 검색 쪽
    (search_similar_chunks 등)은 이 필터와 무관하게 이전 버전(실제 내용)을
    계속 검색 대상으로 쓴다 -- 화면 삭제가 RAG 코퍼스에는 전파되지 않는
    의도된 설계(사용자 방침: "RAG는 지식 로그라 쌓여있는 게 좋다").

    `after`/`before`(2026-10-14 추가): 분할 로딩(get_recent_session_boundary
    참고)을 위한 선택적 시간 범위 필터. 실측해보니(2026-10-14) `ROW_NUMBER()`
    중복제거 + `_ACTIVE_SESSION_FILTER`(행마다 서브쿼리)가 전체 테이블
    기준으로 계산되는 게 진짜 병목이었음 -- 반환되는 행 수를 줄여도 이
    계산 자체를 안 줄이면 쿼리 시간이 거의 안 줄어든다(실측: 32건만 반환
    해도 1초 가까이 걸림, 반면 이 계산 자체를 뺀 단순 필터는 0.2초). 그래서
    아래 두 경우를 구분한다:
    - **`after`만 있고 `before`는 없음(최근 창, 빠른 경로)**: CTE 안쪽
      (중복제거 전)에서부터 시간 필터를 걸어도 안전하다 -- 어떤 줄의 진짜
      최신 버전이 `created_at >= after`라면, 그보다 더 이전(= 더 작은
      created_at)인 다른 버전들은 전부 "더 과거"일 뿐이라 자동으로 같은
      조건을 만족하지 못해 제외되고, 그 줄의 더 나중 버전은 존재할 수
      없으므로(나중 버전은 정의상 더 큰 created_at이라 역시 `>= after`를
      만족) 입력을 좁혀도 rn=1 결과가 달라지지 않는다. 그래서 이 경우만
      입력 자체를 좁혀 무거운 계산 비용을 크게 줄인다.
    - **`before`가 있는 경우(배경 과거분 로딩, 느려도 되는 경로)**: 입력을
      `created_at < before`로 미리 좁히면 위험하다 -- 어떤 줄의 더 오래된
      버전은 이 범위 안에 있는데 진짜 최신 버전은 `before` 이후(이미 "최근
      창"으로 따로 보여준 부분)에 있는 경우, 좁혀진 범위 안에서는 그 최신
      버전의 존재를 모르니 오래된 버전을 최신인 것처럼 잘못 뽑아버릴 수
      있다. 이 경로는 어차피 체감 속도에 영향이 없는 백그라운드 로딩이라,
      안전한(느린) 전체 계산 후 바깥쪽 필터 방식을 그대로 유지한다.
    """
    pool = get_pool()
    course_filter = "AND course_id = %(course_id)s" if course_id else ""
    params: dict = {"course_id": course_id}
    inner_time_filter = ""
    outer_time_filter = ""
    if after is not None and before is None:
        # 빠른 경로(최근 창). `sim:`(재생 테스트) 데이터는 여기서 원천 제외한다
        # -- 2026-10-17 실측: 6차 사이클 재생이 실제 삽입 시각 기준 10/16~17에
        # 전부 몰려있어서, `created_at >= after` 조건만으로는 경계를 아무리
        # 조정해도 사이클테스트 2,303줄이 거의 그대로 "최근 창"에 다시
        # 잡힘(get_recent_session_boundary의 단위 집계에서만 sim을 뺀 걸로는
        # 안 풀리는 문제였음 -- 그건 경계값 계산만 바꿀 뿐, 여기 이 조건은
        # 그대로 실제 시각으로만 걸렀기 때문). 재생 테스트 데이터는 실제
        # 사람이 지금 쓰는 세션이 아니므로, 아래 else 분기(배경 로딩)로 항상
        # 보내도록 여기서 명시적으로 뺀다.
        inner_time_filter = " AND created_at >= %(after)s AND origin_ref NOT LIKE 'sim:%%'"
        params["after"] = after
    else:
        if after is not None:
            outer_time_filter += " AND created_at >= %(after)s"
            params["after"] = after
        if before is not None:
            # 위 빠른 경로에서 뺀 `sim:` 데이터가 여기(배경 로딩)로 전부
            # 들어오도록, 시간 조건과 OR로 묶는다 -- 그래야 "실제 삽입 시각은
            # 최근이지만 사이클테스트라 빠른 경로에서 빠진" 행들이 어느
            # 쪽에서도 안 빠지고 정확히 한 번씩만 로딩된다.
            outer_time_filter += " AND (created_at < %(before)s OR origin_ref LIKE 'sim:%%')"
            params["before"] = before
    query = f"""
        WITH all_notes AS (
            SELECT content, tier2_result, tier2_sources, created_at, client_line_id, origin_ref, session_id,
                   inserted_out_of_place,
                   ROW_NUMBER() OVER (
                       PARTITION BY client_line_id ORDER BY created_at DESC
                   ) AS rn
            FROM note_chunks
            WHERE source = 'personal'
              AND medium != 'paper_ocr'
              {course_filter}
              {_ACTIVE_SESSION_FILTER}
              {inner_time_filter}
        )
        SELECT content, tier2_result, tier2_sources, created_at, client_line_id, origin_ref, session_id,
               inserted_out_of_place
        FROM all_notes
        WHERE (client_line_id IS NULL OR rn = 1) AND content != '' {outer_time_filter}
        ORDER BY created_at ASC;
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            columns = [c.name for c in cur.description]
            return [dict(zip(columns, row)) for row in cur.fetchall()]


def list_sessions(course_id: str | None) -> list[dict]:
    """노트 내보내기(2026-10-18) "세션 선택" 모드용 -- 이 노트의 세션 목록을
    시작시각 오래된순으로 반환(session_id, started_at, line_count).

    날짜 범위(`created_at` 기준) 내보내기는 재생 테스트로 대량 백필된 데이터엔
    못 쓴다는 한계가 있어서(그 데이터는 실제 역사적 날짜와 무관하게 전부
    이번 주 안에 삽입됨, api_export_notes 참고) -- 날짜와 무관하게 "이
    세션(하루 분량)"을 통째로 고를 수 있는 대안으로 추가. get_all_notes()와
    같은 필터(personal만, paper_ocr/빈내용 제외, 삭제된 세션 제외) +
    **같은 중복제거**(client_line_id별 최신본만)를 똑같이 적용해야 한다 --
    처음엔 "집계(COUNT)라 어느 버전이 잡히든 영향 없다"고 생각했는데 실측
    해보니 틀렸음(2026-10-18): COUNT(*)는 같은 줄을 고쳐 쓴 과거 버전까지
    전부 세서, 편집이 많았던 세션(예: 2026-10-08, UI 개발 중 여러 번
    수정/병합/분할 테스트했던 날)은 실제 내보내기 결과(줄 수)보다 훨씬 큰
    숫자를 보여줬다(실측: 두 세션 합쳐 11로 표시됐는데 실제 내보내기는 4줄).
    """
    pool = get_pool()
    course_filter = "AND course_id = %(course_id)s" if course_id else ""
    query = f"""
        WITH ranked AS (
            SELECT session_id, created_at, client_line_id,
                   ROW_NUMBER() OVER (
                       PARTITION BY client_line_id ORDER BY created_at DESC
                   ) AS rn
            FROM note_chunks
            WHERE source = 'personal' AND medium != 'paper_ocr' AND content != ''
              AND session_id IS NOT NULL
              {course_filter}
              {_ACTIVE_SESSION_FILTER}
        )
        SELECT session_id, MIN(created_at) AS started_at, count(*) AS line_count
        FROM ranked
        WHERE client_line_id IS NULL OR rn = 1
        GROUP BY session_id
        ORDER BY started_at ASC;
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


def get_deleted_sessions(
    course_id: str | None, after: datetime | None = None, before: datetime | None = None
) -> list[dict]:
    """이 과목에서 삭제된 세션 묘비를 반환 -- loadAllNotes()가 필기 타임라인에
    "N번째 필기세션 삭제됨" 구분선을 끼워 넣는 데 쓴다.

    `after`/`before`(2026-10-14): get_all_notes()와 같은 분할 로딩 경계를
    묘비 쪽에도 똑같이 적용 -- 안 그러면 초기(최근) 창에서 "최근 세션들
    사이에 지워진 세션"이 빠지거나, 배경(과거) 로딩 때 같은 묘비가 중복으로
    다시 나타날 수 있다. `session_started_at`(삭제 시각이 아니라 그 세션이
    원래 시작됐던 시각) 기준으로 자른다 -- get_all_notes()가 note_chunks의
    `created_at`으로 자르는 것과 같은 축.
    """
    pool = get_pool()
    course_filter = "AND course_id = %(course_id)s" if course_id else ""
    params: dict = {"course_id": course_id}
    time_filter = ""
    if after is not None:
        time_filter += " AND session_started_at >= %(after)s"
        params["after"] = after
    if before is not None:
        time_filter += " AND session_started_at < %(before)s"
        params["before"] = before
    query = f"""
        SELECT day_key, ordinal, session_started_at
        FROM deleted_sessions
        WHERE true {course_filter} {time_filter}
        ORDER BY session_started_at ASC;
    """
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
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
            top_match_feedback_days_ago, session_id, tier2_sources, inserted_out_of_place,
            upload_label
        ) VALUES (
            %(content)s, %(source)s, %(medium)s, %(course_id)s,
            %(note_timestamp)s, %(origin_ref)s,
            %(top_score)s, %(is_question)s, %(is_abbreviated)s, %(is_new_concept)s,
            %(session_duplicate_score)s, %(routing_decision)s, %(tier2_result)s, %(embedding)s::vector,
            %(client_line_id)s, %(hint_category)s, %(top_match_feedback_value)s,
            %(top_match_feedback_days_ago)s, %(session_id)s, %(tier2_sources)s, %(inserted_out_of_place)s,
            %(upload_label)s
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


def get_upload_labels() -> list[str]:
    """지금까지 실제로 쓰인 업로드 분류명(`upload_label`)을 중복 없이 반환.

    2026-10-12: 사용자 지적 -- "분류명을 매번 새로 입력하게 두면 같은 뜻인데
    표현만 다른 값들이 생길 수 있으니, 기존에 쓰인 값을 먼저 보여줘서 재사용을
    유도하는 게 낫다". 프론트엔드가 드롭다운을 열 때 이걸 불러와서 localStorage
    목록과 합친다 -- 그래야 다른 브라우저/기기에서 이미 쓴 분류명도 그대로
    보임(지금까지는 브라우저별 localStorage에만 있어서 기기마다 따로 놀았음).
    course_id로 안 좁히는 이유: 분류명은 "어떤 문서 종류인가"(강의자료/실습자료
    등) 개념이라 과목보다 상위 축이라고 보는 게 맞음.
    """
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT upload_label FROM note_chunks
                WHERE upload_label IS NOT NULL AND upload_label != ''
                ORDER BY upload_label;
                """
            )
            return [row[0] for row in cur.fetchall()]

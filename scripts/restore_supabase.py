"""scripts/backup_supabase.py가 만든 로컬 JSON 백업을 Postgres DB에 다시
채워 넣는다(2026-10-14, Supabase egress 경고 계기로 추가한 재해복구 대비).

**순서**: 1) 대상 DB(새 Supabase 프로젝트든 로컬 Postgres든)에
scripts/schema.sql을 먼저 실행해서 빈 테이블을 만들어둔다. 2) 이 스크립트를
실행한다. 자세한 사용법은 scripts/README.md 참고.

안전장치:
- 대상 DB는 .env의 POSTGRES_URI를 그대로 쓴다 -- 실수로 엉뚱한 DB에 쓰는
  일을 줄이려고, 실행 전 그 연결 문자열의 호스트/DB이름을 화면에 보여주고
  "yes" 입력을 받는다(--yes로 생략 가능, 자동화 용도).
- note_chunks에 이미 행이 있으면(= 복구 대상이 "빈 DB"가 아니라 이미 쓰고
  있는 DB로 보임) 기본적으로 중단한다 -- --force로 강제 진행 가능하지만,
  그래도 각 테이블 INSERT는 전부 자연키 기준 `ON CONFLICT DO NOTHING`이라
  기존 행을 덮어쓰진 않는다(중복만 조용히 건너뜀).
- id 값은 백업에 있던 원본 그대로 넣고, 끝나면 각 테이블의 시퀀스를
  "지금까지 들어간 최대 id + 1"로 맞춰서 이후 새로 들어가는 행이 id가
  겹치지 않게 한다.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

from src.db import get_pool  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"

# 테이블별 컬럼 순서, 자연키(ON CONFLICT 대상), id 시퀀스 이름(없으면 None).
# note_chunks.embedding은 JSON엔 그냥 리스트로 들어있어서 삽입 시
# `%s::vector`로 명시 캐스팅해야 한다(2026-09-29 CLAUDE.md 결정과 같은 이유
# -- psycopg가 list를 double precision[]로 직렬화해버리는 문제).
_TABLES = [
    {
        "name": "note_chunks",
        "conflict": "origin_ref",
        "sequence": "note_chunks_id_seq",
        "columns": [
            "id", "content", "source", "medium", "course_id", "note_timestamp",
            "origin_ref", "top_score", "is_question", "is_abbreviated", "is_new_concept",
            "session_duplicate_score", "routing_decision", "embedding", "created_at",
            "tier2_result", "client_line_id", "feedback_value", "feedback_at",
            "hint_category", "top_match_feedback_value", "top_match_feedback_days_ago",
            "session_id", "tier2_sources", "inserted_out_of_place", "upload_label",
        ],
        "embedding_column": "embedding",
    },
    {
        "name": "deleted_sessions",
        "conflict": "session_id",
        "sequence": None,
        "columns": ["session_id", "course_id", "day_key", "ordinal", "session_started_at", "deleted_at"],
    },
    {
        "name": "failed_ingestions",
        "conflict": "id",
        "sequence": "failed_ingestions_id_seq",
        "columns": ["id", "origin_ref", "error_type", "error_message", "created_at"],
    },
    {
        "name": "period_summaries",
        "conflict": "course_id, period_type, period_key",
        "sequence": "period_summaries_id_seq",
        "columns": ["id", "course_id", "period_type", "period_key", "summary", "note_count", "tier2_count", "created_at"],
    },
    {
        "name": "session_events",
        "conflict": "id",
        "sequence": "session_events_id_seq",
        "columns": ["id", "session_id", "event_type", "note_count", "created_at"],
    },
    {
        "name": "token_usage",
        "conflict": "id",
        "sequence": "token_usage_id_seq",
        "columns": ["id", "created_at", "call_type", "model", "purpose", "input_tokens", "output_tokens"],
    },
]


def _latest_backup_file(table: str) -> Path | None:
    """`backup_<table>_<YYYY-MM-DD>.json` 형태만 정확히 매칭한다.

    단순 glob(`backup_{table}_*.json`)을 썼다가 실측 버그 발견(2026-10-14):
    `data/`에 과거(2026-10-06) 수동 백업 중 다른 명명 규칙을 쓴 파일
    (`backup_note_chunks_all_2026-10-06.json`)이 섞여 있었는데, glob이
    이것도 `backup_note_chunks_*.json`에 매칭시키고 문자열 정렬상
    "_all_..."이 "_2026-10-14"보다 뒤로 와서 "최신 백업"으로 잘못 뽑혔다
    -- 그 결과 더 오래된(지금 스키마의 여러 컬럼이 아직 없던 시절) 백업으로
    복구를 시도해서 INSERT가 실패했다. 테이블명 뒤에 정확히 ISO 날짜만
    오는 패턴으로 제한해서 이런 변칙 파일명을 제외한다.
    """
    pattern = re.compile(rf"^backup_{re.escape(table)}_(\d{{4}}-\d{{2}}-\d{{2}})\.json$")
    matches = [p for p in _DATA_DIR.glob(f"backup_{table}_*.json") if pattern.match(p.name)]
    if not matches:
        return None
    return max(matches, key=lambda p: pattern.match(p.name).group(1))


def restore_table(spec: dict) -> int:
    """백업 JSON을 읽어 대상 테이블에 INSERT.

    행 하나당 `execute()`를 한 번씩 부르면(처음 구현, 2026-10-14 실측) 원격
    Supabase 왕복 지연(건당 0.5~1초, 이 프로젝트 네트워크 환경에서 실측됨)
    때문에 note_chunks(2,700여 행) 하나만 복구하는 데도 수십 분이 걸려서
    실용성이 없었다 -- 여러 행을 하나의 다중 VALUES INSERT로 묶어서 왕복
    횟수 자체를 줄인다(2,700행 -> 배치 50개면 왕복 약 55번).
    embedding 컬럼이 있는 테이블(note_chunks)은 한 배치당 전송량이 커서
    배치를 작게(50), 없는 테이블은 크게(500) 잡는다.
    """
    backup_path = _latest_backup_file(spec["name"])
    if backup_path is None:
        print(f"  (백업 파일 없음, 건너뜀: backup_{spec['name']}_*.json)")
        return 0

    with open(backup_path, encoding="utf-8") as f:
        rows = json.load(f)
    if not rows:
        print(f"  {backup_path.name}: 0행, 건너뜀")
        return 0

    columns = spec["columns"]
    embedding_col = spec.get("embedding_column")
    batch_size = 50 if embedding_col else 500

    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            for start in range(0, len(rows), batch_size):
                chunk = rows[start : start + batch_size]
                value_groups = []
                params: dict = {}
                for i, row in enumerate(chunk):
                    group = []
                    for col in columns:
                        key = f"c{i}_{col}"
                        group.append(f"%({key})s::vector" if col == embedding_col else f"%({key})s")
                        params[key] = row.get(col)
                    value_groups.append("(" + ", ".join(group) + ")")
                query = (
                    f"INSERT INTO {spec['name']} ({', '.join(columns)}) "
                    f"VALUES {', '.join(value_groups)} "
                    f"ON CONFLICT ({spec['conflict']}) DO NOTHING;"
                )
                cur.execute(query, params)
            if spec["sequence"]:
                cur.execute(
                    f"SELECT setval(%s, COALESCE((SELECT MAX(id) FROM {spec['name']}), 1))",
                    (spec["sequence"],),
                )
        conn.commit()
    print(f"  {backup_path.name}: {len(rows)}행 복구 시도 완료")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="로컬 JSON 백업을 Postgres에 복구")
    parser.add_argument("--yes", action="store_true", help="확인 프롬프트 없이 바로 진행")
    parser.add_argument("--force", action="store_true", help="note_chunks에 이미 행이 있어도 진행")
    args = parser.parse_args()

    # inet_server_addr()는 Supabase Session Pooler를 거치면 풀러 자체의
    # 내부 IP를 반환해서(실측 2026-10-14) "어느 프로젝트인지" 알아보기
    # 어렵다 -- .env의 POSTGRES_URI에서 바로 뽑은 호스트/DB이름(예:
    # aws-0-ap-southeast-2.pooler.supabase.com)을 같이 보여준다.
    parsed = urlparse(os.environ["POSTGRES_URI"])
    readable_target = f"{parsed.hostname}:{parsed.port}{parsed.path}"

    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT current_database();")
            db_name = cur.fetchone()[0]
            cur.execute("SELECT to_regclass('note_chunks');")
            table_exists = cur.fetchone()[0] is not None
            existing_count = 0
            if table_exists:
                cur.execute("SELECT count(*) FROM note_chunks;")
                existing_count = cur.fetchone()[0]

    if not table_exists:
        print("note_chunks 테이블이 없습니다 -- scripts/schema.sql을 먼저 이 DB에 실행하세요.")
        sys.exit(1)

    print(f"복구 대상: {readable_target} (database={db_name})")
    print(f"note_chunks 기존 행 수: {existing_count}")
    if existing_count > 0 and not args.force:
        print("이미 데이터가 있는 DB입니다 -- 잘못된 대상에 복구할 위험을 줄이려고 중단합니다.")
        print("정말 이 DB로 진행하려면 --force를 붙여서 다시 실행하세요.")
        sys.exit(1)

    if not args.yes:
        answer = input("이 DB에 복구를 진행할까요? (yes 입력) > ").strip().lower()
        if answer != "yes":
            print("취소했습니다.")
            sys.exit(0)

    for spec in _TABLES:
        print(f"[{spec['name']}]")
        restore_table(spec)

    print("복구 완료.")


if __name__ == "__main__":
    main()

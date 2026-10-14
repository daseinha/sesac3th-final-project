"""Supabase(Postgres)의 모든 테이블을 로컬 JSON으로 백업.

2026-10-14: Supabase 무료 플랜 egress 경고(월 5GB 중 20.88GB 사용, 15.88GB
초과) 계기로 추가. pg_dump가 이 컴퓨터에 안 깔려 있어서 psycopg로 직접 덤프.
egress 초과가 바로 데이터 삭제로 이어지진 않지만(공식 Billing FAQ 기준
알림->유예기간->서비스 제한 순서), 만약을 대비한 안전망 + 데이터 손실 걱정
없이 egress 원인을 나중에 분석할 로컬 사본 확보 목적.

실행: `uv run python scripts/backup_supabase.py`
결과: data/backup_<table>_<YYYY-MM-DD>.json (이미 있던 수동 백업과 같은 명명 규칙)
"""

import json
from datetime import date, datetime
from pathlib import Path
from uuid import UUID

from dotenv import load_dotenv

load_dotenv()

from src.db import get_pool  # noqa: E402

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DATA_DIR = _PROJECT_ROOT / "data"
_TABLES = [
    "note_chunks",
    "deleted_sessions",
    "failed_ingestions",
    "period_summaries",
    "session_events",
    "token_usage",
]


def _json_default(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    # pgvector 컬럼(embedding)은 psycopg가 기본적으로 문자열이나 Vector류
    # 객체로 반환할 수 있음 -- 리스트로 변환 가능하면 그대로 저장.
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def backup_table(table: str) -> int:
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"SELECT * FROM {table};")
            columns = [c.name for c in cur.description]
            rows = [dict(zip(columns, row)) for row in cur.fetchall()]

    out_path = _DATA_DIR / f"backup_{table}_{date.today().isoformat()}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, default=_json_default)
    return len(rows)


if __name__ == "__main__":
    for table in _TABLES:
        count = backup_table(table)
        print(f"{table}: {count}행 백업 완료")

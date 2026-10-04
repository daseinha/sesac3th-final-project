"""날짜 피드(sim:YYYYMMDD) 기준 일자별 tier2 / store_only 추세를 집계한다.

발표용 지표 1번(일자별 store_only 추세). 파이프라인이나 판단 규칙은 건드리지 않고,
이미 DB에 쌓인 판정 결과를 읽기만 한다.

사용: uv run python src/store_only_trend.py
"""

import re
import sys

from dotenv import load_dotenv

from src.db import get_pool

_SIM_DATE_RE = re.compile(r"^sim:(\d{8}):")


def daily_trend() -> list[dict]:
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT origin_ref, routing_decision
                FROM note_chunks
                WHERE source = 'personal' AND origin_ref LIKE 'sim:%'
                """
            )
            rows = cur.fetchall()

    per_day: dict[str, dict] = {}
    for origin_ref, decision in rows:
        m = _SIM_DATE_RE.match(origin_ref or "")
        if not m:
            continue
        day = m.group(1)
        stat = per_day.setdefault(day, {"day": day, "total": 0, "tier2": 0})
        stat["total"] += 1
        if decision == "tier2":
            stat["tier2"] += 1

    result = []
    for day in sorted(per_day):
        stat = per_day[day]
        stat["store_only"] = stat["total"] - stat["tier2"]
        stat["store_only_ratio"] = round(stat["store_only"] / stat["total"], 3) if stat["total"] else None
        result.append(stat)
    return result


def main() -> None:
    load_dotenv()
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"{'date':<10}{'total':>7}{'tier2':>7}{'store':>7}{'store%':>9}")
    for s in daily_trend():
        pct = f"{s['store_only_ratio'] * 100:.1f}%" if s["store_only_ratio"] is not None else "-"
        print(f"{s['day']:<10}{s['total']:>7}{s['tier2']:>7}{s['store_only']:>7}{pct:>9}")


if __name__ == "__main__":
    main()

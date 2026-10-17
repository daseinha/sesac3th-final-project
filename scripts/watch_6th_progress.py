"""6차 사이클 날짜별 완료를 실시간 감지해서 완료되는 즉시 한 줄씩 출력한다.

watch_5th_progress.py와 완전히 동일한 로직, course_id만 "sesac_bootcamp"(실서비스
기본 코퍼스)로 교체. origin_ref가 'sim:%'인 행만 세므로 참고자료/실사용
라이브 데이터와 섞여 있어도 안전하게 6차 재생분만 집계한다.

실행: uv run python -m scripts.watch_6th_progress
(Monitor 도구로 이 명령을 돌려서 "DAY_DONE ..." 줄이 나올 때마다 알림을 받는다)
"""

import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.db import get_pool  # noqa: E402
from src.simulate_day import fine_grained_units  # noqa: E402

COURSE_ID = "sesac_bootcamp"  # 실서비스 기본 코퍼스 -- 다른 값으로 바꾸지 말 것
DATA_DIR = Path("data/personal")
_DAY_FILE_RE = re.compile(r"^\d{8}.*\.txt$")
POLL_SECONDS = 20


def main() -> None:
    files = sorted(p for p in DATA_DIR.glob("*.txt") if _DAY_FILE_RE.match(p.name))
    expected: dict[str, int] = {}
    for p in files:
        date = p.stem[:8]
        text = open(p, encoding="utf-8").read()
        expected[date] = len(fine_grained_units(text))

    reported: set[str] = set()
    pool = get_pool()
    print(f"감시 시작: {len(files)}개 날짜 추적 중 (20초 간격 폴링)", flush=True)

    while True:
        with pool.connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT substring(origin_ref from 'sim:(\\d{8}):'), routing_decision
                    FROM note_chunks
                    WHERE course_id = %s AND origin_ref LIKE 'sim:%%';
                    """,
                    (COURSE_ID,),
                )
                rows = cur.fetchall()

        counts: dict[str, int] = {}
        tier2_counts: dict[str, int] = {}
        for date, decision in rows:
            counts[date] = counts.get(date, 0) + 1
            if decision == "tier2":
                tier2_counts[date] = tier2_counts.get(date, 0) + 1

        for date in sorted(expected):
            if date in reported:
                continue
            exp = expected[date]
            actual = counts.get(date, 0)
            if exp > 0 and actual == exp:
                t2 = tier2_counts.get(date, 0)
                so = actual - t2
                print(f"DAY_DONE {date} total={actual} tier2={t2} store_only={so}", flush=True)
                reported.add(date)

        if len(reported) >= len(expected):
            print("ALL_DAYS_DONE", flush=True)
            break
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()

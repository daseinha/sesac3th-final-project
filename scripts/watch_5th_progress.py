"""5차 사이클 날짜별 완료를 실시간 감지해서 완료되는 즉시 한 줄씩 출력한다.

2026-10-15, 사용자 지적으로 발견한 문제 해결용: 재생 스크립트(replay_5th_cycle.py)의
stdout을 `tail -f`로 직접 보려 했으나, Windows에서 Python stdout이 파일로
리다이렉트되면 전체 버퍼링(line-buffered가 아니라 block-buffered)이 걸려서
수십 줄이 쌓여야 flush되는 바람에, "날짜 완료마다 즉시 알림"이 거의 작동하지
않았음. 재생 스크립트의 출력을 보는 대신, **DB 상태를 직접 주기적으로 조회**해서
"이 날짜가 방금 완료됐다"를 판정하고 그 즉시 print(flush=True)로 이벤트를
내보내는 방식으로 우회한다 -- OpenAI 호출 없이 순수 DB count 쿼리만 반복하므로
비용/부하 거의 없음.

완료 판정 기준은 scripts/resume_5th_cycle.py와 동일: fine_grained_units()로
그 날짜가 몇 줄로 쪼개질지 question_rate=0 조건에서는 결정론적으로 미리 계산
가능하고, DB에 실제 들어간 그 날짜의 줄 수와 비교해서 일치하면 완료로 본다.

실행: uv run python -m scripts.watch_5th_progress
(Monitor 도구로 이 명령을 돌려서 "DAY_DONE ..." 줄이 나올 때마다 알림을 받는다)
"""

import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.db import get_pool  # noqa: E402
from src.simulate_day import fine_grained_units  # noqa: E402

COURSE_ID = "sesac_bootcamp_5th"
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

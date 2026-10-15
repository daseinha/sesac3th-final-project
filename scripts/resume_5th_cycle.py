"""5차 사이클 재생 재개 스크립트 -- 중간에 멈춘 뒤 이어서 할 때 쓴다.

2026-10-15, 사용자가 교육장에서 작업 중이라 긴 재생(예상 5.5~6시간)을 2시간
정도 돌리고 자리를 비워야 하는 상황 -- 나중에 이어서 할 수 있어야 함.

simulate_day.py의 _feed_units()가 짓는 origin_ref는
"sim:{날짜}:{인덱스}:{재생을 실제로 돌린 시각}" 형식이라 타임스탬프 성분이
실행마다 달라진다 -- 그래서 scripts/replay_5th_cycle.py를 중단 지점과 무관하게
그냥 다시 실행하면, UNIQUE 제약(origin_ref)이 타임스탬프가 달라 전혀 안
걸리고, 이미 처리한 날짜까지 전부 새로운 origin_ref로 또 쌓인다(중복).

판정 기준: fine_grained_units()로 그 날짜 파일이 몇 줄로 쪼개질지는
question_rate=0(랜덤 주입 없음)이라 완전히 결정론적으로 미리 계산 가능하다.
DB에 실제로 들어간 그 날짜의 줄 수(origin_ref의 "sim:{날짜}:" 접두사로 식별,
참고자료 복제분은 이 접두사가 없어 섞이지 않음)와 비교해서:
- 일치(그리고 >0)하면: 이미 완료된 날 -> 건너뜀
- 0건이면: 아직 손 안 댄 날 -> 처음부터 처리
- 그 사이(중간에 끊겨서 일부만 들어간 날)면: 그 날짜의 기존 행을 전부
  삭제한 뒤 처음부터 다시 처리 -- 반쪼가리 상태로 남으면 나중 분석에서
  "이 날은 일부만 들어갔다"를 구분할 방법이 없어지므로, 부분본을 남기지
  않고 통째로 다시 만드는 쪽을 택함(해당 날짜는 API 호출이 중복되지만
  57일 중 많아야 1일뿐이라 비용은 무시할 만함).

실행: uv run python -m scripts.resume_5th_cycle
(이전 replay_5th_cycle_result.log에 이어쓰기(append)한다)
"""

import re
import time
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.adapters import _FILENAME_DATE_RE  # noqa: E402
from src.db import get_pool  # noqa: E402
from src.simulate_day import _feed_units, fine_grained_units  # noqa: E402

COURSE_ID = "sesac_bootcamp_5th"
DATA_DIR = Path("data/personal")
LOG_PATH = Path("replay_5th_cycle_result.log")

_DAY_FILE_RE = re.compile(r"^\d{8}.*\.txt$")


def _stored_count(target_date: str) -> int:
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM note_chunks WHERE course_id = %s AND origin_ref LIKE %s;",
                (COURSE_ID, f"sim:{target_date}:%"),
            )
            return cur.fetchone()[0]


def _delete_partial(target_date: str) -> int:
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM note_chunks WHERE course_id = %s AND origin_ref LIKE %s;",
                (COURSE_ID, f"sim:{target_date}:%"),
            )
            deleted = cur.rowcount
            conn.commit()
            return deleted


def main() -> None:
    files = sorted(p for p in DATA_DIR.glob("*.txt") if _DAY_FILE_RE.match(p.name))
    print(f"대상 파일 {len(files)}개. 이어서 처리 시작(course_id={COURSE_ID!r}).")

    total_notes = 0
    total_tier2 = 0
    total_store_only = 0
    total_skipped_days = 0
    log_lines: list[str] = []
    start_time = time.time()

    for day_idx, path in enumerate(files, start=1):
        match = _FILENAME_DATE_RE.match(path.stem)
        target_date = match.group(1) if match else path.stem
        text = open(path, encoding="utf-8").read()
        units = fine_grained_units(text)
        expected = len(units)

        stored = _stored_count(target_date)
        if expected > 0 and stored == expected:
            total_skipped_days += 1
            continue
        if 0 < stored < expected:
            deleted = _delete_partial(target_date)
            print(
                f"[{day_idx}/{len(files)}] {path.name}: 부분처리 발견"
                f"({stored}/{expected}) -- {deleted}건 삭제 후 재처리"
            )

        session_id = str(uuid.uuid4())
        try:
            report = _feed_units(
                units, target_date, COURSE_ID, session_id,
                delay_seconds=0, question_rate=0,
            )
        except Exception as e:  # noqa: BLE001
            print(f"[{day_idx}/{len(files)}] {path.name}: 실패 -- {type(e).__name__}: {e}")
            continue

        total_notes += report["total"]
        total_tier2 += report["tier2"]
        total_store_only += report["store_only"]
        for i, r in enumerate(report["results"]):
            log_lines.append(
                f"[{path.name}:{i}] routing={r['routing_decision']} "
                f"top_score={r['top_score']} is_question={r['is_question']} "
                f"content={r['content'][:60]!r}"
            )

        print(
            f"[{day_idx}/{len(files)}] {path.name}: {report['total']}건 "
            f"(tier2={report['tier2']}, store_only={report['store_only']})"
        )

    elapsed = time.time() - start_time
    summary = (
        f"\n=== 5차 재생 재개 구간 완료 ===\n"
        f"이번 구간에서 건너뛴(이미 완료된) 날: {total_skipped_days}개\n"
        f"이번 구간에서 새로 처리한 줄: {total_notes}건 "
        f"(tier2={total_tier2}, store_only={total_store_only})\n"
        f"소요 시간: {elapsed/60:.1f}분\n"
        f"완료 시각: {datetime.now().isoformat(timespec='seconds')}\n"
    )
    print(summary)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(summary)
        f.write("\n".join(log_lines))
    print(f"상세 로그(이어쓰기): {LOG_PATH}")


if __name__ == "__main__":
    main()

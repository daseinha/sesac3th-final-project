"""6차 사이클 재생 재개 스크립트 -- 중간에 멈춘 뒤 이어서 할 때 쓴다.

resume_5th_cycle.py와 완전히 동일한 로직, course_id만 "sesac_bootcamp"(실서비스
기본 코퍼스)로 교체. 안전성 근거: 삭제/건너뛰기 판정이 전부
`origin_ref LIKE 'sim:{날짜}:%'` 패턴에 의존하는데, 참고자료(file-path 기반
origin_ref)나 실사용 라이브 데이터(live:/ask: 접두사)는 이 패턴에 전혀 안
걸리므로, 같은 course_id 안에 다른 데이터가 섞여 있어도 이 스크립트는
`sim:`로 시작하는 행만 건드린다.

판정 기준(5차와 동일): fine_grained_units()로 그 날짜 파일이 몇 줄로 쪼개질지는
question_rate=0이라 결정론적으로 미리 계산 가능. DB에 실제로 들어간 그 날짜의
줄 수와 비교해서 일치하면 건너뛰고, 0이면 처음부터, 중간이면 그 날짜분만
삭제 후 재처리.

실행: uv run python -m scripts.resume_6th_cycle
(이전 replay_6th_cycle_result.log에 이어쓰기)
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

COURSE_ID = "sesac_bootcamp"  # 실서비스 기본 코퍼스 -- 다른 값으로 바꾸지 말 것
DATA_DIR = Path("data/personal")
LOG_PATH = Path("replay_6th_cycle_result.log")

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
        f"\n=== 6차 재생 재개 구간 완료 ===\n"
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

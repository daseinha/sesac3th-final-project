"""6차 전체 재생 테스트 -- gpt-6-luna 기반, 실서비스 기본 코퍼스(course_id=
"sesac_bootcamp")에 직접 구축한다.

2026-10-17 경위: 3~5차는 전부 gpt-4.1-mini 기준이었고, 격리된 course_id에서
테스트한 뒤 결과가 괜찮으면 나중에 기본 코퍼스로 승격하는 방식이었다(5차가
그 전례). 이번엔 사용자 판단으로 그 절차를 생략하고 처음부터 바로 기본
코퍼스에 구축하기로 함 -- "최신 경량모델이 더 나아질 거라고 예상되니, 끝나고
바꾸는 것보다 지금 바꿔서 시작하는 게 낫다, 기존 사이클들은 백업돼 있으니"
(사용자). 서비스 화면을 매번 다른 주소로 띄울 필요도 없어지는 부수 효과.

**실행 전 전제(실행 스크립트가 직접 검증하지 않음, 사람이 먼저 확인했음)**:
- .env의 LLM_MODEL=gpt-6-luna, LLM_TEMPERATURE=(빈 문자열) 확인됨
- DB 전체 백업 완료(scripts.backup_supabase, 2026-10-17)
- course_id="sesac_bootcamp"의 기존 5차 필기(2,303줄, sim: 접두사)는
  "sesac_bootcamp_5th_archive"로 이름만 바꿔 보존(삭제 아님) -- 이 스크립트는
  그 작업이 이미 끝난 상태에서 실행되는 걸 전제로 함(실행 전 course_id별
  행 수를 다시 확인할 것).
- 참고자료(강의자료 133 + 종이필기 OCR 61)는 이미 "sesac_bootcamp"에 있어서
  별도 복제 불필요.
- 실사용 라이브 타이핑 데이터(live:/ask: origin_ref)는 건드리지 않음 --
  이 스크립트는 origin_ref가 `sim:`으로 시작하는 행만 다룸.

**방법론은 3차/5차와 동일**(replay_5th_cycle.py 그대로 재사용, course_id만
교체) -- simulate_day.py의 feed_day_file(), 줄 단위 청킹, delay_seconds=0,
question_rate=0. 하루 = 세션 한 단위로 session_id 새로 발급.

origin_ref 충돌 위험 없음: `sim:{날짜}:{인덱스}:{재생을 실제로 돌린 시각}`
형식이라 이번 실행 시각(2026-10-17경)이 이전 모든 실행(10/7, 10/8, 10/15,
10/16)과 달라서 UNIQUE 제약과 무관하게 자연히 구분됨.

실행: uv run python -m scripts.replay_6th_cycle
"""

import re
import time
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.simulate_day import feed_day_file  # noqa: E402

COURSE_ID = "sesac_bootcamp"  # 실서비스 기본 코퍼스에 직접 구축 -- 다른 값으로 바꾸지 말 것
DATA_DIR = Path("data/personal")
LOG_PATH = Path("replay_6th_cycle_result.log")

_DAY_FILE_RE = re.compile(r"^\d{8}.*\.txt$")


def main() -> None:
    files = sorted(p for p in DATA_DIR.glob("*.txt") if _DAY_FILE_RE.match(p.name))
    print(
        f"대상 파일 {len(files)}개 발견. course_id={COURSE_ID!r}(실서비스 기본)로 "
        f"재생 시작 (gpt-6-luna, 3차/5차와 동일 조건: 줄 단위 청킹, "
        f"delay_seconds=0, question_rate=0)."
    )

    total_notes = 0
    total_tier2 = 0
    total_store_only = 0
    total_failed_files = 0
    log_lines: list[str] = []
    start_time = time.time()

    for day_idx, path in enumerate(files, start=1):
        session_id = str(uuid.uuid4())
        try:
            report = feed_day_file(
                str(path),
                course_id=COURSE_ID,
                session_id=session_id,
                delay_seconds=0,
                question_rate=0,
            )
        except Exception as e:  # noqa: BLE001 -- 한 파일(하루) 실패해도 나머지는 계속
            print(f"[{day_idx}/{len(files)}] {path.name}: 실패 -- {type(e).__name__}: {e}")
            total_failed_files += 1
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
        f"\n=== 6차 재생 완료 (gpt-6-luna) ===\n"
        f"course_id: {COURSE_ID}\n"
        f"처리한 줄: {total_notes}건 (tier2={total_tier2}, store_only={total_store_only})\n"
        f"처리 실패 파일: {total_failed_files}개\n"
        f"소요 시간: {elapsed/60:.1f}분\n"
        f"완료 시각: {datetime.now().isoformat(timespec='seconds')}\n"
    )
    print(summary)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write(summary)
        f.write("\n".join(log_lines))
    print(f"상세 로그: {LOG_PATH}")


if __name__ == "__main__":
    main()

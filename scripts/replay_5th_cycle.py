"""5차 전체 재생 테스트 -- 3차와 똑같은 방식(simulate_day.py, 줄 단위 청킹)으로
57일치 개인 필기를 새 course_id로 재생한다.

2026-10-15 경위: 4차(course_id="sesac_bootcamp_4th")가 parse_txt_notes()(빈
줄 기준 문단 쪼갬)를 썼는데, 사후 검증 결과 1~3차는 전부 simulate_day.py
(줄바꿈마다 쪼갬)로 돌았다는 게 뒤늦게 드러남 -- 4차는 "실제 서비스 어댑터
동작"을 보여주는 유효한 결과지만, 3차와 순수 비교(같은 조건에서 라우팅 규칙
변경 전후만 비교)를 하려면 청킹 방법 자체를 3차와 똑같이 맞춘 결과가 하나
더 필요함. 이 스크립트가 그 "5차" -- `src/simulate_day.py`의 `feed_day_file()`
를 재구현 없이 그대로 재사용(1~3차와 완전히 같은 로직 보장), 단 사용성
테스트용 옵션(지연/질문주입)은 배치 재생에 안 맞으므로 명시적으로 끈다.

- `delay_seconds=0` -- 원래 기본값 4.0초는 "화면에서 반응 눈으로 보기" 용도
  (라이브 데모용), 배치 재생에서는 불필요한 대기일 뿐. 3차도 이 값으로
  돌렸음(CLAUDE.md 2026-10-05/2026-10-08 변경이력 참고 -- "물음표 랜덤주입
  없음"과 같은 맥락으로 delay도 실질적 영향 없는 파라미터).
- `question_rate=0` -- 기본값 0.12(12% 줄에 "?" 랜덤 추가)는 1차에서 썼지만
  3차는 명시적으로 꺼서(question_rate=0) 돌렸음(CLAUDE.md 2026-10-08 "3차
  재생 완료(물음표 랜덤주입 없음)" 항목) -- 3차와 비교하려면 반드시 동일하게
  꺼야 함.
- 참고자료(강의자료/종이필기)는 scripts/duplicate_reference_for_5th_cycle.py로
  새 course_id에 먼저 복제해둔다(이 스크립트 실행 전 필수).
- origin_ref는 simulate_day.py가 자체적으로 짓는 형식
  (`sim:{날짜}:{인덱스}:{재생을 실제로 돌린 시각}`)을 그대로 쓴다 -- 4차처럼
  별도 접미사를 붙이지 않음. 3차/5차 모두 같은 함수가 "재생을 실제로 돌린
  시각"을 타임스탬프 성분에 포함시키므로(3차=2026-10-08 실행, 5차=2026-10-15
  실행), origin_ref가 날짜까지 같아도 이 성분이 달라 UNIQUE 제약 충돌 위험이
  없음(확인됨).
- 하루 = 세션 한 단위로 session_id를 새로 발급(기존 1~4차와 동일 관례).
- feed_day_file() 내부에는 줄 단위 try/except가 없음(3차도 마찬가지였을
  것 -- 같은 함수이므로 "방법론을 똑같이 맞춘다"는 이 스크립트의 목적상
  일부러 추가하지 않음, src/nodes.py의 Tier2 예외 처리가 이미 API 레벨
  오류는 흡수함). 파일(하루) 단위로만 try/except를 둬서, 한 파일에서 예외가
  나도 나머지 날짜는 계속 진행한다.

실행: uv run python -m scripts.replay_5th_cycle
"""

import re
import time
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.simulate_day import feed_day_file  # noqa: E402

COURSE_ID = "sesac_bootcamp_5th"
DATA_DIR = Path("data/personal")
LOG_PATH = Path("replay_5th_cycle_result.log")

# 57개 "회차" txt 파일만 대상 -- 종이필기 스캔 PDF(Scan_*.pdf)는
# scripts/duplicate_reference_for_5th_cycle.py가 이미 복제해뒀으므로 제외.
_DAY_FILE_RE = re.compile(r"^\d{8}.*\.txt$")


def main() -> None:
    files = sorted(p for p in DATA_DIR.glob("*.txt") if _DAY_FILE_RE.match(p.name))
    print(
        f"대상 파일 {len(files)}개 발견. course_id={COURSE_ID!r}로 재생 시작 "
        f"(3차와 동일 조건: 줄 단위 청킹, delay_seconds=0, question_rate=0)."
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
        f"\n=== 5차 재생 완료 ===\n"
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

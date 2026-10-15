"""4차 전체 재생 테스트 -- 57일치 개인 필기를 새 course_id로 순차 재생한다.

CLAUDE.md/외부 리뷰 취합(노트메이트_리뷰종합_ClaudeGemini_2026-10-14.md) 반영 사항:
- 기존 3차 데이터(course_id="sesac_bootcamp")는 전혀 건드리지 않는다 -- 새
  course_id("sesac_bootcamp_4th")로 완전히 격리해서 돌리므로 삭제/백업복원
  자체가 필요 없다(2026-10-15 사용자 결정).
- 강의자료/종이필기 OCR은 scripts/duplicate_reference_for_4th_cycle.py로 이미
  새 course_id에 복제해뒀다(먼저 실행 필수, origin_ref에 ':4thcycle_ref_copy'
  접미사).
- 표준 어댑터(parse_txt_notes)를 그대로 쓰되, origin_ref는 "sim:YYYYMMDD:"
  접두사 + 원본 origin_ref + ':4th' 접미사로 재구성한다(아래 _tagged_origin_ref
  참고) -- 3차의 동일 파일/인덱스 origin_ref와 충돌(UNIQUE 제약 + ON CONFLICT
  DO NOTHING으로 조용히 무시되는 함정)을 피하면서, 동시에 프론트엔드가
  날짜 구분선을 note_timestamp 기준으로 정확히 그리게 한다(아래 2026-10-15
  수정 메모 참고). 접두사/접미사를 벗기면 3차와 "같은 파일의 같은 줄"을
  그대로 매칭할 수 있어서, 사후 줄 단위 대조(3차 vs 4차)에도 그대로 쓸 수
  있는 키가 된다.
- 하루 = 세션 한 단위로 session_id를 새로 발급(기존 1~3차 운용 방식과 동일,
  2026-09-30 "세션 기능 실사용 데이터 검증" 참고).
- Tier2 예외 처리(2026-10-15 추가, src/nodes.py)가 이제 있어서 API 일시 오류로
  스크립트 전체가 죽지 않지만, 혹시 모를 다른 예외(DB 연결 등)에 대비해 파일
  단위로도 try/except를 둬서 한 파일이 실패해도 나머지 파일은 계속 진행한다.
- 결과는 UTF-8 로그 파일(Windows 콘솔 codepage가 한글을 깨뜨리므로)과 함께,
  사용자가 요청한 대로(test:: 태그 요청에 대한 답변) 이 재생 자체는 실제 데이터라
  dry_run을 쓰지 않는다 -- dry_run은 "57일 역사 재생" 이후 보충 질문을 추가로
  주입할 때만 쓸 것(그건 별도 스크립트).

실행: uv run python -m scripts.replay_4th_cycle
"""

import re
import time
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from src.adapters import parse_txt_notes  # noqa: E402
from src.api import ingest_note  # noqa: E402

COURSE_ID = "sesac_bootcamp_4th"
ORIGIN_SUFFIX = ":4th"
DATA_DIR = Path("data/personal")
LOG_PATH = Path("replay_4th_cycle_result.log")

# 2026-10-15 수정(사용자 지적, 1~2일차까지 처리된 뒤 발견) -- 처음 버전은
# origin_ref를 "{3차와 동일한 경로}:4th"로만 지었는데, 프론트엔드의 날짜
# 구분선 로직(notemate_mockup.html의 noteUnitKey())이 "sim:YYYYMMDD:" 접두사가
# 없으면 note_timestamp(필기 속 원래 날짜)가 아니라 created_at(지금 재생을
# "실제로" 돌리는 시각)으로 날짜를 묶는다 -- 57일치가 전부 "오늘 하루"로
# 뭉쳐 보이게 됨(화면 모니터링만의 문제, SQL 분석에는 영향 없음 --
# note_timestamp 자체는 항상 정확히 저장됨). simulate_day.py가 이미 쓰던
# "sim:YYYYMMDD:..." 접두사를 붙이면 프론트가 그 날짜를 바로 인식해서 제대로
# 구분선을 그려주고, 덤으로 "지난 시간 복습" 버튼(get_sim_day_notes가 같은
# 접두사로 날짜를 찾음)도 이 코스에서 그대로 작동하게 된다.
# "sim:{날짜}:" 다음에 3차와 동일한 원본 origin_ref를 그대로 품게 지어서
# (끝에 ORIGIN_SUFFIX만 붙임), 나중에 "^sim:\d{8}:" 접두사와 ":4th" 접미사만
# 벗기면 3차의 origin_ref와 정확히 같은 문자열이 나온다 -- 줄 단위 대조 키로
# 그대로 재사용 가능.
_SIM_PREFIX_RE = re.compile(r"^sim:(\d{8}):")


def _tagged_origin_ref(original_origin_ref: str, timestamp: datetime | None) -> str:
    date_str = timestamp.strftime("%Y%m%d") if timestamp else "00000000"
    return f"sim:{date_str}:{original_origin_ref}{ORIGIN_SUFFIX}"

# 57개 "회차" txt 파일만 대상 -- 같은 폴더의 종이필기 스캔 PDF(Scan_*.pdf)는
# scripts/duplicate_reference_for_4th_cycle.py가 이미 복제해뒀으므로 제외.
_DAY_FILE_RE = re.compile(r"^\d{8}.*\.txt$")


def main() -> None:
    files = sorted(p for p in DATA_DIR.glob("*.txt") if _DAY_FILE_RE.match(p.name))
    print(f"대상 파일 {len(files)}개 발견. course_id={COURSE_ID!r}로 재생 시작.")

    total_notes = 0
    total_tier2 = 0
    total_store_only = 0
    total_failed_files = 0
    log_lines: list[str] = []
    start_time = time.time()

    for day_idx, path in enumerate(files, start=1):
        session_id = str(uuid.uuid4())
        try:
            notes = parse_txt_notes(str(path))
        except Exception as e:  # noqa: BLE001 -- 파일 하나 파싱 실패해도 나머지는 계속
            print(f"[{day_idx}/{len(files)}] {path.name}: 파싱 실패 -- {type(e).__name__}: {e}")
            total_failed_files += 1
            continue

        day_tier2 = 0
        day_store_only = 0
        for i, note in enumerate(notes):
            origin_ref = _tagged_origin_ref(note.origin_ref, note.timestamp)
            try:
                state = ingest_note(
                    content=note.content,
                    source="personal",
                    medium="txt",
                    origin_ref=origin_ref,
                    course_id=COURSE_ID,
                    timestamp=note.timestamp,
                    session_id=session_id,
                )
            except Exception as e:  # noqa: BLE001 -- 한 줄 실패해도 나머지 줄은 계속
                log_lines.append(f"[{path.name}:{i}] FAILED -- {type(e).__name__}: {e}")
                continue

            total_notes += 1
            decision = state.get("routing_decision")
            if decision == "tier2":
                day_tier2 += 1
                total_tier2 += 1
            elif decision == "store_only":
                day_store_only += 1
                total_store_only += 1
            log_lines.append(
                f"[{path.name}:{i}] origin_ref={origin_ref} "
                f"timestamp={state.get('note_timestamp')} "
                f"top_score={state.get('top_score')} "
                f"routing={decision} "
                f"is_question={(state.get('assessment') or {}).get('is_question')}"
            )

        print(
            f"[{day_idx}/{len(files)}] {path.name}: {len(notes)}건 "
            f"(tier2={day_tier2}, store_only={day_store_only})"
        )

    elapsed = time.time() - start_time
    summary = (
        f"\n=== 4차 재생 완료 ===\n"
        f"course_id: {COURSE_ID}\n"
        f"처리한 줄: {total_notes}건 (tier2={total_tier2}, store_only={total_store_only})\n"
        f"파싱 실패 파일: {total_failed_files}개\n"
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

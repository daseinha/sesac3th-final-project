"""하루치 필기를 줄/의미 단위로 쪼개서, 실시간 타이핑을 흉내 내며 순차 투입하는 스크립트.

2026-10-02: 사용자가 합본 파일을 날짜별 개별 파일로 직접 쪼개서
`data/personal/`에 넣어주기로 함 -- `feed_day_file()`이 그 개별 파일을 받는
주 진입점. 합본 파일에서 날짜 구간만 골라 추출하는 `feed_day()`/
`extract_day_block()`도 혹시 필요할 때를 대비해 남겨둠(개별 파일이 아직
준비 안 됐을 때의 대안).

사용성 테스트 + 성능추이 분석을 겸한다는 요구사항에 맞춰:
- 각 줄 사이에 지연(delay_seconds)을 둬서, 브라우저에서 "필기 입력 -> 큐 생성"
  반응을 눈으로 확인할 시간을 준다(화면은 src/server.py가 띄운 페이지가
  3초마다 폴링해서 자동으로 반영함, notemate_mockup.html 참고)
- 끝나면 그날의 Tier1/Tier2 라우팅 통계를 요약해서 돌려준다 -- 사용자가
  리뷰하는 동안 참고할 정량 지표
- 날짜(파일) 하나만 처리하고 멈춘다 -- 다음 날짜로 넘어가는 건 호출하는
  쪽(대화)이 명시적으로 다시 호출하는 방식으로 제어(자동으로 전체를
  밀어붙이지 않음)
"""

import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

# Windows에서 stdout을 파일로 리다이렉트하면 콘솔 코드페이지(cp949)로
# 떨어져서, 실제 필기에 흔히 나오는 "•" 같은 유니코드 문자를 print()하다가
# UnicodeEncodeError로 스크립트 전체가 죽는다(2026-10-02, 8일차 피드가
# 5/34줄만 처리된 채 멈춘 원인). stdout을 UTF-8로 강제해서 근본적으로 막는다.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.adapters import _DAY_HEADER_RE, _FILENAME_DATE_RE, _has_substantial_content
from src.api import ingest_note

_SEPARATOR_LINE_RE = re.compile(r"^-{5,}$")


def inject_pseudo_questions(
    units: list[str], rate: float = 0.12, seed: int | None = None
) -> tuple[list[str], list[int]]:
    """일부 줄 끝에 "?"를 붙여 질문형 필기를 흉내낸다.

    2026-10-02 사용자 요청: 실제 필기 데이터엔 물음표로 끝나는 문장이 거의
    없어서(평소 습관상 문장을 ?로 잘 안 끝냄) is_question 오버라이드 경로가
    실전 데이터로는 거의 안 걸림 -- 테스트가 다양한 사용 시나리오를 반영하게
    일부 줄을 질문형으로 섞는다. 이미 ?로 끝나거나 너무 짧은 줄(헤더성 등)은
    대상에서 제외. 반환값의 두 번째 요소(바뀐 줄의 인덱스)는 리뷰 보고에 쓰려고
    남겨둠 -- 실제 원문엔 없던 변형이니 어떤 줄이 바뀌었는지 투명하게 밝혀야 함.
    """
    rng = random.Random(seed)
    candidates = [i for i, u in enumerate(units) if len(u) >= 6 and not u.rstrip().endswith(("?", "？"))]
    n = round(len(candidates) * rate)
    chosen = sorted(rng.sample(candidates, min(n, len(candidates))))
    modified = list(units)
    for i in chosen:
        modified[i] = modified[i].rstrip() + "?"
    return modified, chosen


def extract_day_block(text: str, target_date: str) -> str:
    """합본 텍스트에서 target_date(YYYYMMDD)에 해당하는 구간만 잘라낸다.

    해당 날짜의 헤더부터, 다음 날짜 헤더가 나오기 전까지(또는 파일 끝까지).
    """
    lines = text.split("\n")
    start = None
    end = len(lines)
    for i, line in enumerate(lines):
        m = _DAY_HEADER_RE.match(line.strip())
        if m:
            if m.group(1) == target_date:
                start = i
            elif start is not None:
                end = i
                break
    if start is None:
        raise ValueError(f"'{target_date}' 날짜 헤더를 합본 파일에서 못 찾음")
    return "\n".join(lines[start:end])


def fine_grained_units(day_block: str) -> list[str]:
    """하루 구간을 줄 단위(빈 줄 문단이 아니라 한 줄씩)로 쪼갠다.

    실시간 타이핑 시뮬레이션 목적 -- parse_txt_notes()의 문단(빈 줄) 단위보다
    훨씬 잘게, 실제로 한 줄씩 입력하는 상황을 흉내낸다. 날짜 헤더 줄과
    구분선(---) 줄, 내용 없는 줄은 건너뛴다(기존 검증된 필터 재사용).
    """
    units = []
    for line in day_block.split("\n"):
        line = line.strip()
        if not line:
            continue
        if _DAY_HEADER_RE.match(line):
            continue
        if _SEPARATOR_LINE_RE.match(line):
            continue
        if not _has_substantial_content(line):
            continue
        units.append(line)
    return units


def _feed_units(
    units: list[str],
    target_date: str,
    course_id: str,
    session_id: str | None,
    delay_seconds: float,
    question_rate: float = 0.12,
    question_seed: int | None = None,
) -> dict:
    """줄 단위 목록을 순서대로 실시간처럼 투입하고 라우팅 통계를 반환.

    타임스탬프는 원본 날짜가 아니라 "지금"을 씀 -- 과거 데이터를 복원하는
    게 아니라, 그 내용을 "오늘 실시간으로 타이핑하는 중"이라고 가정하는
    사용성 테스트이기 때문(프론트가 '오늘' 기준으로 폴링하므로 이래야
    화면에 보임).

    question_rate: 일부 줄 끝에 "?"를 붙여 질문형 필기를 섞는다(기본 12%,
    0으로 주면 끔) -- inject_pseudo_questions() 참고. 실제 원문을 왜곡하는
    거라 어떤 줄이 바뀌었는지 results/리턴값에 투명하게 남긴다.
    """
    session_id = session_id or f"sim-{target_date}"
    units, injected_indices = inject_pseudo_questions(units, rate=question_rate, seed=question_seed)
    injected_set = set(injected_indices)

    results = []
    for i, content in enumerate(units):
        state = ingest_note(
            content=content,
            source="personal",
            medium="txt",
            origin_ref=f"sim:{target_date}:{i}:{datetime.now().isoformat()}",
            course_id=course_id,
            timestamp=datetime.now(),
            session_id=session_id,
        )
        results.append({
            "content": content,
            "synthetic_question": i in injected_set,
            "routing_decision": state.get("routing_decision"),
            "top_score": state.get("top_score"),
            "is_question": state.get("is_question"),
            "tier2_result": state.get("tier2_result"),
        })
        tag = " [Q!]" if i in injected_set else ""
        print(f"[{i+1}/{len(units)}]{tag} {state.get('routing_decision')} -- {content[:40]}")
        if i < len(units) - 1:
            time.sleep(delay_seconds)

    tier2_count = sum(1 for r in results if r["routing_decision"] == "tier2")
    return {
        "target_date": target_date,
        "total": len(results),
        "tier2": tier2_count,
        "store_only": len(results) - tier2_count,
        "tier2_ratio": round(tier2_count / len(results), 3) if results else None,
        "injected_question_count": len(injected_indices),
        "results": results,
    }


def feed_day_file(
    path: str,
    course_id: str = "sesac_bootcamp",
    session_id: str | None = None,
    delay_seconds: float = 4.0,
    question_rate: float = 0.12,
    question_seed: int | None = None,
) -> dict:
    """날짜별 개별 필기 파일(사용자가 직접 쪼개서 제공) 하나를 줄 단위로 투입.

    파일명 앞 8자리(YYYYMMDD)를 target_date로 쓴다 -- 기존 `parse_txt_notes()`
    파일명 규칙과 동일. 파일 안에 날짜 헤더 줄이 남아있어도(그대로 복붙한
    경우) `fine_grained_units()`가 걸러낸다.
    """
    match = _FILENAME_DATE_RE.match(Path(path).stem)
    target_date = match.group(1) if match else Path(path).stem
    text = open(path, encoding="utf-8").read()
    units = fine_grained_units(text)
    return _feed_units(units, target_date, course_id, session_id, delay_seconds, question_rate, question_seed)


def feed_day(
    combined_path: str,
    target_date: str,
    course_id: str = "sesac_bootcamp",
    session_id: str | None = None,
    delay_seconds: float = 4.0,
    question_rate: float = 0.12,
    question_seed: int | None = None,
) -> dict:
    """합본 파일에서 target_date(YYYYMMDD) 구간만 추출해 줄 단위로 투입.

    개별 날짜 파일이 아직 준비 안 됐을 때의 대안 경로(주 경로는
    `feed_day_file()`).
    """
    text = open(combined_path, encoding="utf-8").read()
    day_block = extract_day_block(text, target_date)
    units = fine_grained_units(day_block)
    return _feed_units(units, target_date, course_id, session_id, delay_seconds, question_rate, question_seed)


if __name__ == "__main__":
    import argparse

    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="하루치 필기를 줄 단위로 실시간처럼 투입")
    parser.add_argument("path", help="개별 날짜 파일 경로(주 경로)")
    parser.add_argument("--course-id", default="sesac_bootcamp")
    parser.add_argument("--delay", type=float, default=4.0)
    args = parser.parse_args()

    report = feed_day_file(args.path, args.course_id, delay_seconds=args.delay)
    print(f"\n=== {report['target_date']} 완료 ===")
    print(f"전체 {report['total']}개 -- tier2 {report['tier2']} / store_only {report['store_only']} (tier2 비율 {report['tier2_ratio']})")

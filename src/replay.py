"""재생(replay) 스크립트 -- CLAUDE.md "백엔드 검증 방식" 2번.

txt 필기 파일을 시간순으로 파싱해서 ingest_note()에 순차 투입하고, 각 필기의
라우팅 판단/힌트를 로그 파일로 남긴다. 임계값 튜닝, 개인화 효과 확인용.

결과를 print()가 아니라 UTF-8 로그 파일로 남기는 이유: Windows 콘솔 codepage가
한글을 깨뜨려서 눈으로 확인하려면 파일로 열어봐야 함 (VS Code 등에서).
"""

import argparse

from src.adapters import parse_txt_notes
from src.api import ingest_note


def replay(
    path: str,
    course_id: str | None,
    source: str = "personal",
    medium: str = "txt",
    log_path: str = "replay_result.log",
) -> None:
    notes = parse_txt_notes(path)

    with open(log_path, "w", encoding="utf-8") as log:
        for i, note in enumerate(notes, start=1):
            result = ingest_note(
                content=note.content,
                source=source,
                medium=medium,
                origin_ref=note.origin_ref,
                course_id=course_id,
                timestamp=note.timestamp,
            )
            assessment = result.get("assessment") or {}
            log.write(f"=== [{i}/{len(notes)}] {note.origin_ref} ===\n")
            log.write(f"timestamp: {note.timestamp}\n")
            log.write(f"content: {note.content}\n")
            log.write(f"top_score: {result.get('top_score'):.4f}\n")
            log.write(
                "assessment: "
                f"is_question={assessment.get('is_question')}, "
                f"is_abbreviated={assessment.get('is_abbreviated')}, "
                f"is_new_concept={assessment.get('is_new_concept')}\n"
            )
            log.write(f"routing_decision: {result.get('routing_decision')}\n")
            if result.get("tier2_result"):
                log.write(f"tier2_result: {result.get('tier2_result')}\n")
            log.write("\n")

    print(f"{len(notes)} notes replayed. Log written to: {log_path}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Replay a txt note file through ingest_note()")
    parser.add_argument("path", nargs="?", default="src/sample_notes.txt")
    parser.add_argument("--course-id", default="langgraph_basics")
    parser.add_argument("--log", default="replay_result.log")
    args = parser.parse_args()

    replay(args.path, course_id=args.course_id, log_path=args.log)

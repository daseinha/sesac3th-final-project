"""data/ 폴더를 스캔해서 하위 폴더(personal/base/stt_reference)에 따라
알맞은 source로 ingest_file()을 순차 실행하는 스크립트.

테스트/최적화용 -- 실제 필기·수업자료를 data/<source>/ 아래에 넣어두면
파일 하나하나 직접 호출할 필요 없이 이 스크립트 한 번으로 전부 처리된다.
data/는 .gitignore에 등록돼 있어서 여기 넣는 실제 파일은 커밋되지 않는다.
"""

import argparse
from pathlib import Path

from src.api import _FILE_ADAPTERS, ingest_file

_SOURCE_FOLDERS = ("personal", "base", "stt_reference")


def ingest_data_folder(
    data_dir: str = "data",
    course_id: str | None = None,
    log_path: str = "ingest_folder_result.log",
) -> None:
    data_path = Path(data_dir)

    with open(log_path, "w", encoding="utf-8") as log:
        for source in _SOURCE_FOLDERS:
            folder = data_path / source
            if not folder.is_dir():
                continue

            files = [
                f
                for f in sorted(folder.iterdir())
                if f.is_file() and f.suffix.lower() in _FILE_ADAPTERS
            ]
            skipped = [
                f
                for f in sorted(folder.iterdir())
                if f.is_file() and f.suffix.lower() not in _FILE_ADAPTERS and f.name != ".gitkeep"
            ]

            log.write(f"### source={source} ({len(files)}개 파일) ###\n")
            for skipped_file in skipped:
                log.write(f"[스킵 -- 지원 안 하는 확장자] {skipped_file.name}\n")

            for f in files:
                log.write(f"\n--- {f.name} ---\n")
                try:
                    results = ingest_file(str(f), source=source, course_id=course_id)
                except Exception as e:  # noqa: BLE001 -- 파일 하나 실패해도 나머지는 계속
                    log.write(f"[실패] {type(e).__name__}: {e}\n")
                    continue

                log.write(f"{len(results)}개 청크 저장됨\n")
                for r in results:
                    if r.get("routing_decision"):  # personal만 값이 있음
                        log.write(
                            f"  top_score={r.get('top_score'):.4f}, "
                            f"routing_decision={r.get('routing_decision')}\n"
                        )
            log.write("\n")

    print(f"완료. 로그: {log_path}")


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(description="Ingest all files under data/<source>/")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--course-id", default=None)
    parser.add_argument("--log", default="ingest_folder_result.log")
    args = parser.parse_args()

    ingest_data_folder(args.data_dir, course_id=args.course_id, log_path=args.log)

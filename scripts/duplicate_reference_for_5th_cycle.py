"""5차 사이클(줄 단위 재청킹, 3차와 동일 조건) 준비 -- 강의자료/종이필기를
새 격리 course_id로 복제한다.

2026-10-15 경위: 4차(course_id="sesac_bootcamp_4th")가 parse_txt_notes()(빈
줄 기준 문단 쪼갬)를 썼는데, 1~3차는 실제로 simulate_day.py(줄바꿈마다 쪼갬)로
돌았다는 걸 뒤늦게 발견 -- 청킹 단위 자체가 달라서 3차와 순수 비교가 안 됐음
(내용은 전혀 안 빠졌고 "묶음 단위"만 달랐음). 5차는 simulate_day.py와 동일한
줄 단위 청킹으로 3차와 조건을 정확히 맞춘 비교군. 4차 결과는 지우지 않고 그대로
둔다 -- 그 나름대로 "실제 서비스 어댑터가 어떻게 동작하는지" 보여주는 유효한
결과라서 보존.
"""

from dotenv import load_dotenv

load_dotenv()

from src.db import get_pool  # noqa: E402

SOURCE_COURSE_ID = "sesac_bootcamp"
TARGET_COURSE_ID = "sesac_bootcamp_5th"
ORIGIN_SUFFIX = ":5thcycle_ref_copy"

_COLUMNS = (
    "content, source, medium, course_id, note_timestamp, origin_ref, top_score, "
    "is_question, is_abbreviated, is_new_concept, session_duplicate_score, "
    "routing_decision, embedding, tier2_result, client_line_id, hint_category, "
    "top_match_feedback_value, top_match_feedback_days_ago, session_id, "
    "tier2_sources, inserted_out_of_place, upload_label"
)


def main() -> None:
    pool = get_pool()
    with pool.connection() as conn:
        with conn.cursor() as cur:
            # 복제 전 안전 확인: 이미 복제된 적 있으면 중복 실행 방지(멱등성 확인용 로그만)
            cur.execute(
                "SELECT count(*) FROM note_chunks WHERE course_id = %(cid)s;",
                {"cid": TARGET_COURSE_ID},
            )
            existing = cur.fetchone()[0]
            if existing:
                print(f"경고: {TARGET_COURSE_ID!r}에 이미 {existing}건 존재. "
                      f"ON CONFLICT DO NOTHING이라 중복 삽입은 안 되지만 확인 바람.")

            query = f"""
                INSERT INTO note_chunks ({_COLUMNS})
                SELECT content, source, medium, %(target_course_id)s, note_timestamp,
                       origin_ref || %(suffix)s, top_score, is_question, is_abbreviated,
                       is_new_concept, session_duplicate_score, routing_decision, embedding,
                       tier2_result, client_line_id, hint_category, top_match_feedback_value,
                       top_match_feedback_days_ago, session_id, tier2_sources,
                       inserted_out_of_place, upload_label
                FROM note_chunks
                WHERE course_id = %(source_course_id)s
                  AND (source = 'base' OR medium = 'paper_ocr')
                ON CONFLICT (origin_ref) DO NOTHING;
            """
            cur.execute(
                query,
                {
                    "target_course_id": TARGET_COURSE_ID,
                    "suffix": ORIGIN_SUFFIX,
                    "source_course_id": SOURCE_COURSE_ID,
                },
            )
            inserted = cur.rowcount
            conn.commit()

            cur.execute(
                "SELECT source, medium, count(*) FROM note_chunks WHERE course_id = %(cid)s "
                "GROUP BY source, medium ORDER BY source, medium;",
                {"cid": TARGET_COURSE_ID},
            )
            print(f"복제 완료: {inserted}건 삽입. course_id={TARGET_COURSE_ID!r} 현재 구성:")
            for row in cur.fetchall():
                print(f"  {row[0]:>14} / {row[1]:<14} {row[2]:>5}건")


if __name__ == "__main__":
    main()

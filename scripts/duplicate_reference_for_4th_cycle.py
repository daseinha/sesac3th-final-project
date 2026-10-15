"""4차 전체 재생 테스트 준비 -- 강의자료/종이필기 OCR을 새 course_id로 복제한다.

CLAUDE.md 확인된 사실: course_id는 "공유 단위"가 아니라 "격리 단위"로 설계/운용돼
왔다(검색 쿼리 전부가 course_id로 필터링, 2026-10-11 베타테스트도 별도 course_id로
의도적 격리에 썼던 전례 있음). 그래서 새 course_id로 4차 개인 필기를 재생하면
강의자료(base)/종이필기(paper_ocr)를 전혀 못 찾는다 -- 이 둘을 새 course_id로
복제해서 "참고 자료 조건"만 3차와 동등하게 맞춘다.

사용자 결정(2026-10-15): 종이필기(61건)도 포함한다 -- 3차 때는 이 데이터가
유실된 상태였어서(10/1 사고 복구 때 같이 날아감, 10/14 복구) 3차는 이 자료를
한 번도 써본 적이 없다. 이건 "규칙 변화"와 별개인 "코퍼스 변화"라 4차 결과
해석 시 분리해서 기록해야 한다(아래 manifest 출력 참고).

임베딩은 이미 계산된 값을 그대로 복사한다 -- OpenAI API 호출 전혀 없음, 순수
SQL INSERT...SELECT라 몇 초면 끝난다.

origin_ref에 ':4thcycle_ref_copy' 접미사를 붙인다 -- origin_ref는 UNIQUE 제약이
course_id와 무관하게 테이블 전체에 걸려 있어서, 접미사 없이 그대로 복사하면
기존 행과 충돌해 ON CONFLICT DO NOTHING으로 조용히 무시된다(실제로 이 함정을
사전에 발견해서 막은 것 -- CLAUDE.md 2026-10-15 참고).
"""

from dotenv import load_dotenv

load_dotenv()

from src.db import get_pool  # noqa: E402

SOURCE_COURSE_ID = "sesac_bootcamp"
TARGET_COURSE_ID = "sesac_bootcamp_4th"
ORIGIN_SUFFIX = ":4thcycle_ref_copy"

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
            # 복제 전 코퍼스 구성 manifest 기록 (3차 vs 4차 해석 시 "규칙 변화"와
            # "코퍼스 변화"를 분리하기 위함, 외부 리뷰 "4차 전체 재생 전 점검" 2번 반영)
            cur.execute(
                "SELECT source, medium, count(*) FROM note_chunks "
                "WHERE course_id = %(cid)s GROUP BY source, medium ORDER BY source, medium;",
                {"cid": SOURCE_COURSE_ID},
            )
            print(f"=== {SOURCE_COURSE_ID} (원본) 코퍼스 구성 ===")
            for row in cur.fetchall():
                print(f"  {row[0]:>14} / {row[1]:<14} {row[2]:>5}건")

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
                "SELECT source, medium, count(*) FROM note_chunks "
                "WHERE course_id = %(cid)s GROUP BY source, medium ORDER BY source, medium;",
                {"cid": TARGET_COURSE_ID},
            )
            print(f"\n=== {TARGET_COURSE_ID} (복제본) 코퍼스 구성 (방금 삽입: {inserted}건) ===")
            for row in cur.fetchall():
                print(f"  {row[0]:>14} / {row[1]:<14} {row[2]:>5}건")


if __name__ == "__main__":
    main()

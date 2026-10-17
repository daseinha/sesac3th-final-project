"""6차(gpt-6-luna) vs 5차(4.1-mini) 최종 비교 분석 -- 57일 전체.

줄 단위 content-매칭 diff 전수조사 + 토큰/비용 전체 합계 + 기간별 추이.
"""
from dotenv import load_dotenv
load_dotenv()
from src.db import get_pool

pool = get_pool()

with pool.connection() as conn:
    with conn.cursor() as cur:
        cur.execute("""
            SELECT distinct substring(origin_ref from 'sim:(\d{8}):') FROM note_chunks
            WHERE course_id='sesac_bootcamp' AND origin_ref LIKE 'sim:%'
        """)
        dates = sorted(r[0] for r in cur.fetchall())

print(f"총 날짜(회차) 수: {len(dates)}")

total6_tier2 = total6_store = total5_tier2 = total5_store = 0
total_lines = 0
mismatches = []
length_mismatch_dates = []

with pool.connection() as conn:
    with conn.cursor() as cur:
        for d in dates:
            cur.execute("""
                SELECT content, routing_decision, top_score, is_question, is_new_concept, is_abbreviated, session_duplicate_score
                FROM note_chunks WHERE course_id='sesac_bootcamp' AND origin_ref LIKE %s ORDER BY origin_ref
            """, (f"sim:{d}:%",))
            six = cur.fetchall()
            cur.execute("""
                SELECT content, routing_decision, top_score, is_question, is_new_concept, is_abbreviated, session_duplicate_score
                FROM note_chunks WHERE course_id='sesac_bootcamp_5th_archive' AND origin_ref LIKE %s ORDER BY origin_ref
            """, (f"sim:{d}:%",))
            five = cur.fetchall()
            if len(six) != len(five):
                length_mismatch_dates.append((d, len(six), len(five)))
                continue
            for a, b in zip(six, five):
                total_lines += 1
                if a[1] == "tier2": total6_tier2 += 1
                else: total6_store += 1
                if b[1] == "tier2": total5_tier2 += 1
                else: total5_store += 1
                if a[0] != b[0]:
                    continue
                if a[1] != b[1]:
                    cause = "other"
                    if a[2] == b[2]:  # top_score 동일
                        if a[4] != b[4]:
                            cause = "is_new_concept_flip"
                        elif a[5] != b[5]:
                            cause = "is_abbreviated_flip"
                        elif a[3] != b[3]:
                            cause = "is_question_flip"
                    else:
                        cause = "top_score_differs(corpus)"
                    direction = "6th_only_tier2" if a[1] == "tier2" else "5th_only_tier2"
                    mismatches.append({"date": d, "content": a[0][:50], "cause": cause, "direction": direction,
                                        "top_score_6": a[2], "top_score_5": b[2]})

print(f"\n길이 불일치 날짜: {length_mismatch_dates}")
print(f"\n비교 가능한 총 줄 수: {total_lines}")
print(f"6차: tier2={total6_tier2} ({total6_tier2/total_lines*100:.1f}%) store_only={total6_store} ({total6_store/total_lines*100:.1f}%)")
print(f"5차: tier2={total5_tier2} ({total5_tier2/total_lines*100:.1f}%) store_only={total5_store} ({total5_store/total_lines*100:.1f}%)")
print(f"\n라우팅 불일치 총 건수: {len(mismatches)} ({len(mismatches)/total_lines*100:.2f}%)")

from collections import Counter
cause_counts = Counter(m["cause"] for m in mismatches)
direction_counts = Counter(m["direction"] for m in mismatches)
print(f"\n원인별: {dict(cause_counts)}")
print(f"방향별: {dict(direction_counts)}")

# 반복 문구 빈도 top
content_counts = Counter(m["content"] for m in mismatches)
print(f"\n가장 자주 등장한 불일치 내용 top 10:")
for content, cnt in content_counts.most_common(10):
    print(f"  {cnt}건: {content!r}")

import json
with open(".scratch/mismatches_full.json", "w", encoding="utf-8") as f:
    json.dump(mismatches, f, ensure_ascii=False)
print(f"\n상세 저장: .scratch/mismatches_full.json")

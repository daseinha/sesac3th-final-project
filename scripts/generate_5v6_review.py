"""5차(4.1-mini) vs 6차(gpt-6-luna) 힌트 비교 '피드백선택평가셋' 생성 (2026-10-17).

data/review_run3_all_notes_2026-10-07.html과 같은 스타일(연속 타임라인,
날짜 헤딩, 줄마다 내용+힌트박스+피드백버튼)을 그대로 따르되, 한 줄에 대해
두 모델이 각각 만든 힌트가 있으면 "힌트 A"/"힌트 B"로 블라인드 배치해서
나란히 보여준다(배치는 줄마다 시드 고정 랜덤). 피드백 버튼은 실제
/api/hint_feedback을 그대로 호출 -- origin_ref가 globally UNIQUE라
course_id와 무관하게 정확한 행에 기록된다.

실행: uv run python -m scripts.generate_5v6_review
"""
import random

from dotenv import load_dotenv

load_dotenv()

from src.db import get_pool  # noqa: E402

pool = get_pool()
random.seed(20261017)

with pool.connection() as conn:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT distinct substring(origin_ref from 'sim:(\\d{8}):') FROM note_chunks "
            "WHERE course_id='sesac_bootcamp' AND origin_ref LIKE 'sim:%'"
        )
        dates = sorted(r[0] for r in cur.fetchall())


def esc(s: str) -> str:
    return (s or "-").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


total_lines = 0
total_both_hint = 0
day_blocks = []

with pool.connection() as conn:
    with conn.cursor() as cur:
        for day_idx, d in enumerate(dates, start=1):
            cur.execute(
                "SELECT content, routing_decision, tier2_result, tier2_sources, origin_ref "
                "FROM note_chunks WHERE course_id='sesac_bootcamp' AND origin_ref LIKE %s ORDER BY origin_ref",
                (f"sim:{d}:%",),
            )
            six = cur.fetchall()
            cur.execute(
                "SELECT content, routing_decision, tier2_result, tier2_sources, origin_ref "
                "FROM note_chunks WHERE course_id='sesac_bootcamp_5th_archive' AND origin_ref LIKE %s ORDER BY origin_ref",
                (f"sim:{d}:%",),
            )
            five = cur.fetchall()
            if len(six) != len(five):
                print(f"[경고] {d}: 6차 {len(six)}줄 vs 5차 {len(five)}줄 -- 길이 불일치, 건너뜀")
                continue

            date_fmt = f"{d[:4]}-{d[4:6]}-{d[6:]}"
            lines_html = []
            for (c6, r6, h6, s6, o6), (c5, r5, h5, s5, o5) in zip(six, five):
                if c6 != c5:
                    continue
                total_lines += 1
                content_esc = esc(c6)
                panels = []

                if h6 and h5:
                    total_both_hint += 1
                    pair = [("6th", h6, s6, o6), ("5th", h5, s5, o5)]
                    if random.random() < 0.5:
                        pair = [pair[1], pair[0]]
                    pair_key = "|".join(sorted([o6, o5]))
                    letters = ["A", "B"]
                    origins_by_letter = {}
                    rows_ab = []
                    for letter, (_model, h, s, o) in zip(letters, pair):
                        origins_by_letter[letter] = o
                        hint_box = (
                            f'<div class="hint-box" data-origin="{o}" data-pair-key="{pair_key}">'
                            f'<div class="hint-label">힌트 {letter}</div>'
                            f'<div class="hint-text">{esc(h)}</div>'
                            f'<div class="hint-sources">근거: {esc(s)}</div>'
                            '<div class="feedback-row">'
                            '<button class="fb-btn fb-plus" data-value="+" title="도움 됨">+</button>'
                            '<button class="fb-btn fb-minus" data-value="&minus;" title="도움 안 됨">&minus;</button>'
                            '<button class="fb-btn fb-check" data-value="&#10003;" title="이해완료">&#10003;</button>'
                            "</div>"
                            '<div class="fb-note">클릭하면 기록돼요</div></div>'
                        )
                        pref_side = (
                            '<div class="pref-side">'
                            f'<button class="pref-choice" data-pair-key="{pair_key}" data-value="{letter}" data-origin="{o}">'
                            f'<span class="pref-dot"></span>힌트 {letter}가<br>더 낫다</button></div>'
                        )
                        rows_ab.append(f'<div class="compare-row">{pref_side}{hint_box}</div>')
                    # Track C 스타일 4지선다(2026-10-17, 두 차례 사용자 지적 반영한 최종 배치):
                    # 힌트 A 옆엔 "A가 낫다"만, 힌트 B 옆엔 "B가 낫다"만 두고, 그 사이
                    # 가운데에 "비슷함"/"둘 다 아쉬움"을 배치 -- 박스와 바로 붙어 있어야
                    # "이게 이 박스에 대한 선택"이라는 게 직관적으로 보임. 선호 표시는
                    # 별도 텍스트 없이 그 힌트박스 자체 테두리 강조(.preferred)로.
                    pref_mid = (
                        '<div class="pref-mid">'
                        f'<button class="pref-choice" data-pair-key="{pair_key}" data-value="similar">'
                        '<span class="pref-dot"></span>비슷함</button>'
                        f'<button class="pref-choice" data-pair-key="{pair_key}" data-value="both_bad">'
                        '<span class="pref-dot"></span>둘 다 아쉬움</button>'
                        "</div>"
                    )
                    panels.append(rows_ab[0] + pref_mid + rows_ab[1])
                elif h6 or h5:
                    h, s, o = (h6, s6, o6) if h6 else (h5, s5, o5)
                    panels.append(
                        f'<div class="hint-box single" data-origin="{o}">'
                        '<div class="hint-label">힌트(한쪽만 생성됨)</div>'
                        f'<div class="hint-text">{esc(h)}</div>'
                        f'<div class="hint-sources">근거: {esc(s)}</div>'
                        '<div class="feedback-row">'
                        '<button class="fb-btn fb-plus" data-value="+" title="도움 됨">+</button>'
                        '<button class="fb-btn fb-minus" data-value="&minus;" title="도움 안 됨">&minus;</button>'
                        '<button class="fb-btn fb-check" data-value="&#10003;" title="이해완료">&#10003;</button>'
                        "</div>"
                        '<div class="fb-note">클릭하면 기록돼요</div></div>'
                    )

                cls = "both-hint" if (h6 and h5) else ("one-hint" if (h6 or h5) else "no-hint")
                block = f'<div class="line-block {cls}"><div class="content">{content_esc}'
                if not h6 and not h5:
                    block += '<span class="routing-tag store-only">둘 다 STORE ONLY</span>'
                block += "</div>" + "".join(panels) + "</div>"
                lines_html.append(block)

            day_blocks.append(
                f'<h2 class="day-heading">{date_fmt} '
                f'<span class="day-count">({day_idx}/{len(dates)}회차, {len(six)}줄)</span></h2>\n'
                + "\n".join(lines_html)
            )

CSS = """
* { box-sizing: border-box; }
body { margin:0; padding:0 0 4rem 0; background:#fff; color:#1a1a1a;
  font-family:"Malgun Gothic","맑은 고딕","Apple SD Gothic Neo","Noto Sans KR",sans-serif; line-height:1.65; }
.wrap { max-width:880px; margin:0 auto; padding:2rem 1.5rem; }
h1 { font-size:1.5rem; margin:0 0 0.25rem 0; }
.subtitle { color:#666; font-size:0.9rem; margin-bottom:1.5rem; }
.summary { background:#f6f7f9; border:1px solid #e3e5e9; border-radius:10px; padding:1rem 1.25rem; margin-bottom:2rem; font-size:0.95rem; }
.summary table { border-collapse:collapse; width:100%; }
.summary td { padding:0.2rem 0.6rem 0.2rem 0; vertical-align:top; }
.summary td.num { font-weight:600; text-align:right; }
h2.day-heading { font-size:1.15rem; margin:2.5rem 0 1rem 0; padding:0.5rem 0.75rem; background:#eef2f7;
  border-left:4px solid #4a6fa5; border-radius:4px; position:sticky; top:0; }
.day-count { color:#777; font-weight:normal; font-size:0.85rem; margin-left:0.5rem; }
.line-block { margin:0 0 1.1rem 0; padding:0 0 0.6rem 0; border-bottom:1px solid #f0f0f0; }
.line-block.no-hint .content { color:#999; }
.line-block.both-hint { background:#fafbfd; border-radius:8px; padding:0.6rem; }
.content { white-space:pre-wrap; font-size:0.98rem; margin:0 0 0.3rem 0; }
.hint-box { background:#fff4e8; border:1px solid #f0d9b5; border-radius:8px; padding:0.6rem 0.85rem;
  margin:0.4rem 0 0.4rem 0; font-size:0.92rem; white-space:pre-wrap; color:#5a3e1b; }
.hint-label { font-size:0.75rem; font-weight:600; color:#b4792b; text-transform:uppercase; letter-spacing:0.03em; margin-bottom:0.25rem; display:block; }
.hint-box.preferred { box-shadow:0 0 0 2px #3457d5; border-color:#3457d5; }
.compare-row { display:flex; gap:10px; align-items:stretch; margin:0.3rem 0; }
.pref-side { display:flex; flex-direction:column; justify-content:center; flex:0 0 76px; }
.pref-mid { display:flex; justify-content:center; gap:18px; margin:0.15rem 0; }
.pref-choice { display:flex; align-items:center; gap:6px; border:none; background:transparent; padding:2px;
  font-size:12px; line-height:1.25; cursor:pointer; font-family:inherit; color:#444; text-align:left; }
.pref-dot { width:15px; height:15px; border-radius:50%; border:1.5px solid #b4b8c2; flex-shrink:0; background:#fff; }
.pref-choice.selected { color:#3457d5; font-weight:600; }
.pref-choice.selected .pref-dot { background:#3457d5; border-color:#3457d5; box-shadow:inset 0 0 0 2.5px #fff; }
.pref-choice[data-value="both_bad"].selected { color:#c23b3b; }
.pref-choice[data-value="both_bad"].selected .pref-dot { background:#c23b3b; border-color:#c23b3b; }
.hint-sources { font-size:0.75rem; color:#8a6a3a; margin-top:0.2rem; }
.routing-tag { display:inline-block; font-size:0.72rem; font-weight:600; padding:0.05rem 0.45rem; border-radius:4px; margin-left:0.4rem; vertical-align:middle; background:#eee; color:#888; }
.feedback-row { display:flex; gap:6px; justify-content:flex-end; margin-top:0.5rem; }
.fb-btn { width:27px; height:27px; border-radius:50%; border:1.5px solid #ddd; cursor:pointer; font-size:16px; font-weight:700; line-height:1; padding:0; background:#fff; }
.fb-btn.fb-plus { color:#1f9d6a; } .fb-btn.fb-minus { color:#c23b3b; } .fb-btn.fb-check { color:#3457d5; }
.fb-btn.selected.fb-plus { background:#1f9d6a; color:#fff; border-color:#1f9d6a; }
.fb-btn.selected.fb-minus { background:#c23b3b; color:#fff; border-color:#c23b3b; }
.fb-btn.selected.fb-check { background:#3457d5; color:#fff; border-color:#3457d5; }
.fb-note { font-size:0.7rem; color:#aaa; text-align:right; margin-top:0.15rem; }
"""

body_html = "".join(day_blocks)
html = (
    "<!doctype html>\n<html lang=\"ko\">\n<head>\n<meta charset=\"utf-8\">\n"
    "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
    "<title>5차 vs 6차 힌트 비교 피드백선택평가셋</title>\n"
    f"<style>{CSS}</style>\n</head>\n<body>\n<div class=\"wrap\">\n"
    "<h1>5차(4.1-mini) vs 6차(gpt-6-luna) 힌트 비교 — 피드백선택평가셋</h1>\n"
    "<div class=\"subtitle\">2026-10-17 -- 같은 필기 줄에 대해 두 모델이 각각 만든 힌트를 "
    "블라인드(힌트 A/B, 매 줄 랜덤배치 -- 어느 모델이 A/B인지는 개별 줄마다 안 드러남)로 "
    "블라인드(힌트 A/B, 매 줄 랜덤배치)로 비교합니다. 한쪽만 힌트를 만든 줄은 그 하나만, "
    "둘 다 store_only면 내용만 표시합니다. 피드백 버튼은 실제 서비스와 같은 "
    "<code>/api/hint_feedback</code>에 기록됩니다(session_id=\"review-5v6-2026-10-17\"로 구분).</div>\n"
    "<div class=\"summary\"><table>\n"
    f"<tr><td>전체 줄 수</td><td class=\"num\">{total_lines:,}</td></tr>\n"
    f"<tr><td>날짜(회차) 수</td><td class=\"num\">{len(dates)}</td></tr>\n"
    f"<tr><td>양쪽 다 힌트 생성됨</td><td class=\"num\">{total_both_hint:,} "
    f"({total_both_hint / total_lines * 100:.1f}%)</td></tr>\n"
    "</table></div>\n"
    f"{body_html}\n</div>\n"
    "<script>\n"
    "  const REVIEW_SESSION_ID = 'review-5v6-2026-10-17';\n"
    "  // 힌트 A/B 옆 버튼과 가운데(비슷함/둘다아쉬움) 버튼이 서로 다른 DOM 컨테이너에\n"
    "  // 나뉘어 있어서(사용자 요청으로 A옆/B옆/가운데 3곳으로 배치), data-pair-key로\n"
    "  // 흩어진 버튼들을 다시 찾아 모아서 상호배타 선택을 적용한다.\n"
    "  function applyPrefChoice(pairKey, value, originA, originB) {\n"
    "    document.querySelectorAll(`.pref-choice[data-pair-key=\"${CSS.escape(pairKey)}\"]`).forEach((b) => {\n"
    "      b.classList.toggle('selected', b.dataset.value === value);\n"
    "    });\n"
    "    const boxA = document.querySelector(`.hint-box[data-origin=\"${CSS.escape(originA)}\"]`);\n"
    "    const boxB = document.querySelector(`.hint-box[data-origin=\"${CSS.escape(originB)}\"]`);\n"
    "    if (boxA) boxA.classList.toggle('preferred', value === 'A');\n"
    "    if (boxB) boxB.classList.toggle('preferred', value === 'B');\n"
    "    // preferred_origin 필드에는 A/B면 실제 origin_ref를, similar/both_bad면 그 문자열 그대로 저장 --\n"
    "    // 복원 시 이 값이 origin_ref와 일치하면 A/B, 아니면 similar/both_bad 버튼을 그대로 다시 선택한다.\n"
    "    const stored = (value === 'A') ? originA : (value === 'B') ? originB : value;\n"
    "    const body = new URLSearchParams({ session_id: REVIEW_SESSION_ID, pair_key: pairKey, preferred_origin: stored });\n"
    "    fetch('/api/model_preference', { method: 'POST', body }).catch(() => {});\n"
    "  }\n"
    "  document.querySelectorAll('.pref-choice').forEach((btn) => {\n"
    "    btn.addEventListener('click', () => {\n"
    "      const pairKey = btn.dataset.pairKey;\n"
    "      // A/B 버튼 자신은 data-origin을 갖고 있고, 가운데(비슷함/둘다아쉬움) 버튼은\n"
    "      // 없으므로 같은 pair_key를 가진 A쪽 버튼에서 origin을 가져온다.\n"
    "      const aBtn = document.querySelector(`.pref-choice[data-pair-key=\"${CSS.escape(pairKey)}\"][data-value=\"A\"]`);\n"
    "      const bBtn = document.querySelector(`.pref-choice[data-pair-key=\"${CSS.escape(pairKey)}\"][data-value=\"B\"]`);\n"
    "      applyPrefChoice(pairKey, btn.dataset.value, aBtn ? aBtn.dataset.origin : null, bBtn ? bBtn.dataset.origin : null);\n"
    "    });\n"
    "  });\n"
    "  fetch('/api/model_preference_state')\n"
    "    .then((res) => res.json())\n"
    "    .then((state) => {\n"
    "      const seenPairs = new Set();\n"
    "      document.querySelectorAll('.pref-choice[data-pair-key]').forEach((btn) => {\n"
    "        const pairKey = btn.dataset.pairKey;\n"
    "        if (seenPairs.has(pairKey)) return;\n"
    "        seenPairs.add(pairKey);\n"
    "        const stored = state[pairKey];\n"
    "        if (!stored) return;\n"
    "        const aBtn = document.querySelector(`.pref-choice[data-pair-key=\"${CSS.escape(pairKey)}\"][data-value=\"A\"]`);\n"
    "        const bBtn = document.querySelector(`.pref-choice[data-pair-key=\"${CSS.escape(pairKey)}\"][data-value=\"B\"]`);\n"
    "        const originA = aBtn ? aBtn.dataset.origin : null;\n"
    "        const originB = bBtn ? bBtn.dataset.origin : null;\n"
    "        let value = stored;\n"
    "        if (stored === originA) value = 'A';\n"
    "        else if (stored === originB) value = 'B';\n"
    "        document.querySelectorAll(`.pref-choice[data-pair-key=\"${CSS.escape(pairKey)}\"]`).forEach((b) => {\n"
    "          b.classList.toggle('selected', b.dataset.value === value);\n"
    "        });\n"
    "        const boxA = document.querySelector(`.hint-box[data-origin=\"${CSS.escape(originA)}\"]`);\n"
    "        const boxB = document.querySelector(`.hint-box[data-origin=\"${CSS.escape(originB)}\"]`);\n"
    "        if (boxA) boxA.classList.toggle('preferred', value === 'A');\n"
    "        if (boxB) boxB.classList.toggle('preferred', value === 'B');\n"
    "      });\n"
    "    })\n"
    "    .catch(() => {});\n"
    "  document.querySelectorAll('.fb-btn').forEach((btn) => {\n"
    "    btn.addEventListener('click', () => {\n"
    "      const box = btn.closest('.hint-box');\n"
    "      const text = box.querySelector('.hint-text').textContent;\n"
    "      box.querySelectorAll('.fb-btn').forEach((b) => b.classList.remove('selected'));\n"
    "      btn.classList.add('selected');\n"
    "      const body = new URLSearchParams({\n"
    "        course_id: 'sesac_bootcamp', session_id: REVIEW_SESSION_ID,\n"
    "        hint_text: text, value: btn.dataset.value, kind: 'hint', origin_ref: box.dataset.origin,\n"
    "      });\n"
    "      fetch('/api/hint_feedback', { method: 'POST', body }).catch(() => {});\n"
    "    });\n"
    "  });\n"
    "  fetch('/api/hint_feedback_state')\n"
    "    .then((res) => res.json())\n"
    "    .then((state) => {\n"
    "      document.querySelectorAll('.hint-box').forEach((box) => {\n"
    "        const value = state[box.dataset.origin];\n"
    "        if (!value) return;\n"
    "        box.querySelectorAll('.fb-btn').forEach((b) => { b.classList.toggle('selected', b.dataset.value === value); });\n"
    "      });\n"
    "    })\n"
    "    .catch(() => {});\n"
    "</script>\n</body></html>\n"
)

with open("data/review_5v6_hint_comparison.html", "w", encoding="utf-8") as f:
    f.write(html)
print(f"생성 완료: {total_lines}줄, 양쪽 다 힌트 {total_both_hint}건, 파일 크기 {len(html):,} bytes")

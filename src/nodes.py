"""LangGraph 노드 함수 모음.

assess_note는 실제 로직(임베딩 + 벡터 검색 + LLM 구조화 출력)으로 구현됐다.
tier2_deep_agent, store_only는 아직 스텁 -- 각 함수의 TODO 참고.
"""

import os
from datetime import datetime, timezone

import requests
from pydantic import BaseModel, Field

from src.db import (
    insert_note_chunk,
    log_failed_ingestion,
    search_similar_chunks,
    search_top_similarity,
)
from src.llm import embed_text, invoke_chat, invoke_structured
from src.retrieval import build_hybrid_retriever
from src.routers import NEGATIVE_FEEDBACK_SUPPRESS_THRESHOLD, compute_routing_decision
from src.session_memory import add_to_session, check_session_duplicate
from src.state import NoteAssessment, NoteMateState

# 힌트 근거 표시(_summarize_tier2_sources)가 "이 매치는 노이즈가 아니라 진짜
# 관련 있다"고 인정하는 유사도 최소값(2026-10-10, 두 번째 수정).
#
# 경위: 처음엔 이 기준이 아예 없어서(벡터 매치면 크기 안 보고 다 인정) 완전히
# 무관한 질문("막걸리 발효 효모", "그리스의 수도" 등)에도 "내 필기 N건
# 참고"가 떴음(사용자가 직접 재현 제보) -- 벡터 검색이 항상 top-k를 반환하는
# 특성 때문에 유사도가 아무리 낮아도 뭔가는 걸림. 그래서 다시 크기 기준이
# 필요해졌는데, 바로 전에 썼던 라우팅용 LOW_THRESHOLD(0.75)를 재사용하면
# 안 된다는 것도 바로 전 수정에서 실측으로 확인됨(Tier2의 가장 흔한 트리거
# 자체가 "top_score<0.75"라서, 그 값을 그대로 쓰면 가장 흔한 경로에서 항상
# "근거 없음"이 됨).
#
# 그래서 완전히 새로운, 더 낮은 값을 실제 유사도 분포로 다시 잡았다 -- 1차
# 로는 0.35를 썼는데, 검증 중 무관한 질문("콜로세움 수용 인원", "라면 면발
# 탄력")의 최고 유사도가 0.33~0.35까지 올라오는 경우를 발견해서(노이즈
# 바닥이 생각보다 높음 -- 아래 참고) 다시 0.40으로 올림. 지금까지 모은
# 샘플(무관한 질문 7개 최고유사도 0.26~0.3501, 실제 관련 있는 질문 5개
# 최고유사도 0.4055~0.6327) 전부가 0.3501과 0.4055 사이에서 깨끗하게
# 갈려서 0.40을 그 틈 사이로 선택. [잠정] 그래도 표본이 10여 건뿐인 가벼운
# 실측 근거 -- 나중에 실사용 데이터가 쌓이면 재검증 필요(LOW_THRESHOLD/
# HIGH_THRESHOLD와 같은 성격의 미확정 값). 참고로 검증 중 눈치챈 것: 사용자가
# 이 기능을 테스트하면서 무관한 질문들을 `test::` 태그 없이 필기창에 그대로
# 입력해서, 그 질문들 자체가 지금 note_chunks에 진짜 personal 필기로 쌓이고
# 있음 -- 서로 비슷한 "일부러 무관하게 고른" 질문끼리 유사도가 올라가는
# 자기강화 노이즈가 생길 수 있어, 이후 이런 탐색적 테스트는 `test::`를 쓰는
# 게 나음(CLAUDE.md 변경 이력에 기록).
SOURCE_RELEVANCE_FLOOR = 0.40


class _NoteAssessmentSchema(BaseModel):
    """LLM 구조화 출력 스키마. NoteAssessment(TypedDict)와 필드를 동일하게 맞춘다."""

    is_question: bool = Field(
        description="필기에 명시적인 질문(물음표, '~가 뭐지' 같은 표현)이 포함되어 있는가"
    )
    is_abbreviated: bool = Field(
        description="필기가 너무 축약돼 있어 나중에 다시 봤을 때 이해하기 어려운가"
    )
    is_new_concept: bool = Field(
        description="필기가 이전에 다루지 않은 새로운 개념을 담고 있는가"
    )


# _ABBREVIATED_WORD_GUARD 이상이면 is_abbreviated를 LLM 판단과 무관하게 False로
# 강제한다(2026-10-05, 57일 전체 재검토 -- is_abbreviated가 전체의 70%에서
# True로 찍혀서 거의 상시 켜져 있었음). 문장부호 유무는 가드 조건에서 제외함
# -- "※n8n도 서버다." 처럼 문장부호가 있어도 진짜 축약인 사례가 있어서,
# 그 기준을 쓰면 실제 축약 메모를 놓침. 어절 수만 기준으로 삼는다.
_ABBREVIATED_WORD_GUARD = 5

_ASSESSMENT_SYSTEM_PROMPT = (
    "당신은 학습자의 실시간 필기를 분석하는 어시스턴트입니다. "
    "주어진 필기 한 조각을 보고 아래 세 가지를 판단하세요.\n"
    "1. is_question: 명시적인 질문을 담고 있는가\n"
    "2. is_abbreviated: 축약이 심해 나중에 봤을 때 이해하기 어려운가\n"
    "   - 아닌 예(False로 판단할 것): 완결된 개조식 문장. 예: \"CASE문은 조건별 분기 처리 시 사용\", "
    "\"nvm은 Node.js 버전 매니저의 약자로, 여러 버전을 설치해 선택적으로 쓸 수 있게 해준다\". "
    "서술어가 명사형으로 끝나거나 조사가 생략돼도, 의미가 완결돼 있으면 축약이 아님.\n"
    "   - 맞는 예(True로 판단할 것): \"※n8n도 서버다.\"처럼 짧고 핵심만 적어서 전후 맥락 없이는 "
    "무슨 뜻인지 추측이 필요한 메모, 단어나 구만 덩그러니 적힌 경우.\n"
    "3. is_new_concept: 새로운 개념을 다루는가\n"
    "필기 원문 외의 맥락은 알 수 없으니, 주어진 텍스트만 보고 판단하세요."
)


def _resolve_search_course_ids(state: NoteMateState) -> list[str] | None:
    """노트 묶음(RAG 공유그룹, 2026-10-17 야간) -- 검색 범위를 결정한다.

    `search_course_ids`가 있으면 그대로 쓰고(묶음, 항상 자기 자신 포함해서
    프론트가 계산해 보냄), 없으면 `course_id` 하나짜리 리스트로 대체해서
    기존 동작(단일 노트, 묶음 없음)과 완전히 동일하게 만든다.
    """
    search_ids = state.get("search_course_ids")
    if search_ids:
        return search_ids
    course_id = state.get("course_id")
    return [course_id] if course_id else None


def assess_note(state: NoteMateState) -> dict:
    """정리 노드: 신호 A(top_score) + 신호 B(assessment) 계산.

    - 신호 A: 임베딩 생성 -> note_chunks에서 코사인 유사도 검색.
      비교할 기존 데이터가 없으면(첫 필기 등) top_score=0.0으로 처리한다
      ("지원 자료 없음"은 확신도가 낮은 것과 동일하게 취급 -> 자연히 tier2로 라우팅됨).
    - 신호 B: LLM 구조화 출력으로 필기 완결도 판단.
    - 세션 내 작업기억 중복 검사: session_id가 있으면 이번 세션에서 이미 다룬
      내용과 얼마나 비슷한지 확인 -> session_duplicate_score. session_id가
      없으면(호출하는 쪽이 안 넘겼으면) None -- 세션 개념을 안 쓰는 호출도 지원.

    TODO: 세션 내 작업기억의 "용도2(긴 세션 맥락 압축)"는 아직 미구현
    """
    content = state["content"]

    embedding = embed_text(content, purpose="assess_note_embedding")

    top_match = search_top_similarity(embedding, course_id=_resolve_search_course_ids(state))
    top_score = top_match["similarity"] if top_match else 0.0
    # 힌트 피드백 반영 Phase 2(✓ 억제, 2026-10-08): 가장 비슷한 과거 필기가
    # "이미 알고 있음(✓)" 피드백을 받은 적 있으면 그 정보를 같이 들고 간다 --
    # 실제 억제 여부 판단은 routers.compute_routing_decision이 함(세션중복
    # 억제와 같은 자리/패턴).
    top_match_feedback_value = None
    top_match_feedback_days_ago = None
    if top_match and top_match["feedback_value"] and top_match["feedback_at"]:
        top_match_feedback_value = top_match["feedback_value"]
        top_match_feedback_days_ago = (
            datetime.now(timezone.utc) - top_match["feedback_at"]
        ).total_seconds() / 86400
    # 힌트 예외처리(2026-10-18 야간 설계, 2026-10-18 라우팅 연결) -- 가장
    # 비슷한 과거 필기가 "힌트 불필요(템플릿/양식)" 표시돼 있는지. 실제 억제
    # 여부(+유사도 기준) 판단은 ✓ 억제와 같은 자리에서 routers.
    # compute_routing_decision이 함.
    top_match_hint_excluded = bool(top_match and top_match.get("hint_excluded"))

    session_id = state.get("session_id")
    session_duplicate_score, session_duplicate_elapsed_seconds = (
        check_session_duplicate(session_id, embedding) if session_id else (None, None)
    )

    result: _NoteAssessmentSchema = invoke_structured(
        _NoteAssessmentSchema,
        [
            {"role": "system", "content": _ASSESSMENT_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        purpose="assess_note_assessment",
    )
    # 어절 수 가드레일: 일정 길이 이상이면 LLM이 True로 판단했어도 축약이 아닌
    # 것으로 강제한다 -- 긴 완결 문장까지 축약으로 과잉판정되는 걸 막는 결정론적
    # 1차 필터(위 _ABBREVIATED_WORD_GUARD 주석 참고).
    is_abbreviated = result.is_abbreviated
    if len(content.split()) >= _ABBREVIATED_WORD_GUARD:
        is_abbreviated = False

    assessment: NoteAssessment = {
        "is_question": result.is_question,
        "is_abbreviated": is_abbreviated,
        "is_new_concept": result.is_new_concept,
    }

    return {
        "embedding": embedding,
        "top_score": top_score,
        "assessment": assessment,
        "session_duplicate_score": session_duplicate_score,
        "session_duplicate_elapsed_seconds": session_duplicate_elapsed_seconds,
        "top_match_feedback_value": top_match_feedback_value,
        "top_match_feedback_days_ago": top_match_feedback_days_ago,
        "top_match_hint_excluded": top_match_hint_excluded,
    }


def embed_and_store(state: NoteMateState) -> dict:
    """참고자료(source="base"/"stt_reference") 전용 경로 -- 임베딩만 만들고 바로 저장.

    업로드된 강의자료/참고자료는 사용자가 실시간으로 쓰는 필기가 아니라서
    "질문인가/축약됐나" 같은 판단이나 힌트 생성이 의미가 없다 (누구에게 줄
    힌트인지 불분명). Tier1/2 전체를 건너뛰고 여기서 바로 persist_note로 간다 --
    LLM 호출을 아끼고, 판단 로그 통계에 필기가 아닌 자료가 섞이는 것도 방지.
    """
    embedding = embed_text(state["content"], purpose="reference_embedding")
    return {"embedding": embedding}


def decide_routing(state: NoteMateState) -> dict:
    """assess_note 이후, 판단 결과(routing_decision)를 State에 기록하는 노드.

    실제 임계값 로직은 routers.compute_routing_decision()에 있고, 여기선
    그 결과를 State에 저장만 한다 -- 뒤이은 조건부 엣지(routers.route)가
    이 값을 읽어서 다음 노드를 고르고, DB 저장 시 판단 로그 컬럼으로도 남는다.
    """
    return {"routing_decision": compute_routing_decision(state)}


_TIER2_SYSTEM_PROMPT = (
    "당신은 학습자 곁에서 조용히 돕는 학습 동반자입니다. "
    "방금 남긴 필기와, 학습자가 이전에 작성해둔 관련 필기들을 참고해서 "
    "짧고 실질적인 힌트를 하나 작성하세요.\n"
    "- 참고할 이전 필기가 없다면, 그 점을 알리고 필기 자체에 대한 일반적인 도움을 주세요.\n"
    "- 장황하게 설명하지 말고, 카드 한 장에 들어갈 정도로 간결하게 작성하세요."
)

# 힌트가 "어디서 온 답인지" 사용자에게 보여주기 위한 라벨(2026-10-09, 강사님
# 피드백 + 사용자 질문 반영 -- "이 힌트가 RAG인지 LLM 사전학습 지식인지
# 구분해서 보여주면 유용하다"). source 값별 한국어 표기만 담당 -- 실제 개수
# 집계는 아래 _summarize_tier2_sources()가 담당.
_SOURCE_LABELS = {"personal": "내 필기", "base": "강의자료", "stt_reference": "참고자료"}


def _summarize_tier2_sources(similar_docs: list, bm25_origin_refs: set | None = None) -> str:
    """검색된 참고 문서들의 source 구성을 사용자가 읽을 짧은 문구로 요약.

    "비율"(답변 텍스트의 몇 %가 어디서 왔는지)은 LLM이 그렇게 출처를 태깅하며
    글을 쓰지 않아서 측정 자체가 불가능 -- 대신 "실제로 검색돼서 LLM에게
    참고자료로 제공된 문서가 몇 건, 어떤 source였는지"를 보여준다(과장 없는
    정직한 근사치). 검색 결과가 하나도 없으면 LLM이 자기 사전학습 지식으로만
    답한 것이므로 그렇게 표시한다. 웹검색은 Tier2에 아직 없는 기능이라
    ([[deferred-web-search-addon]]) 포함하지 않음 -- 있는 척하면 거짓 표시가 됨.

    2026-10-09 실측 수정: 하이브리드 검색이 유사도와 무관하게 "코퍼스에 뭐든
    있으면 top-k를 무조건 반환"하는 특성 때문에, 완전히 무관한 내용(예: 부트
    캠프와 무관한 "양자컴퓨터 표면코드" 질문)에도 "내 필기 9건"처럼 뜨는 문제를
    실제로 확인함(BM25 쪽 키워드 우연 일치가 대부분). 벡터 쪽엔 유사도 점수가
    있으니(`metadata['similarity']`), BM25 전용 매치(벡터 점수가 없는 것,
    `similarity is None`)는 신뢰할 점수가 없어서 "근거"로 안 센다 -- 이번
    실측에서 바로 이런 매치들이 키워드만 우연히 겹친 가짜 근거였음.

    2026-10-10 1차 수정(철회): 여기에 더해 라우팅용 LOW_THRESHOLD(0.75)를
    "관련성" 기준으로도 재사용해서 유사도가 낮은 매치까지 걸러내려 했었는데,
    실사용 테스트에서 거의 항상 "일반 지식 위주"만 뜨는 버그로 드러남(사용자
    제보: 필기 내용을 그대로 다시 입력해도 똑같이 나옴). 원인은 구조적임 --
    Tier2가 가장 흔하게 걸리는 트리거 자체가 "top_score < 0.75"(신규 개념
    판정)라서, 그 트리거로 들어온 경우 코퍼스 전체에서의 최고 유사도 자체가
    이미 0.75 미만으로 확정돼 있음. 즉 같은 0.75를 "근거 인정 기준"으로도
    쓰면 바로 그 가장 흔한 경로에서 검색된 모든 문서가 자동으로 걸러져서,
    실제로는 참고한 문서가 있어도 항상 "없음"으로 표시됨 -- 사실상 상시
    오작동.

    2026-10-10 2차 수정: 그래서 유사도 크기 기준을 완전히 뺐더니, 반대 방향
    버그가 바로 드러남(사용자가 "막걸리 발효 효모", "그리스의 수도" 같은
    완전히 무관한 질문으로 재현 제보) -- 벡터 검색은 유사도가 아무리 낮아도
    항상 top-k를 반환하므로, 크기를 안 보면 무관한 질문도 전부 "내 필기 N건
    참고"로 뜸(위 2026-10-09 항목과 본질적으로 같은 문제가 벡터 쪽에서도
    재발한 것). 결론: 크기 기준 자체는 필요하지만 0.75는 라우팅 트리거와
    충돌해서 못 쓴다는 것 -- `SOURCE_RELEVANCE_FLOOR`(0.35, 위 선언 참고)로
    새로 분리해서 적용.

    2026-10-14 3차 수정: 0.40으로도 실사용 중 다시 뚫림 -- "오늘 AI 관련
    뉴스", "최근 AI가 푼 수학 난제", "제미나이 최신 모델" 같은 질문들이
    우연히 0.40을 넘겨서 "내 필기 N건 참고"로 표시됐는데, 실제 답변은 전부
    할루시네이션이었음(그날 AI/LLM 관련 필기를 많이 써서 "AI", "모델" 같은
    흔한 단어의 임베딩 공간 근접성만으로 점수가 올라간 것으로 추정). 벡터
    유사도 "크기"만으로는 이 노이즈를 못 거른다는 게 재확인됨 -- 그래서
    기준을 하나 더 추가: 같은 문서를 **BM25(키워드 기반)도 독립적으로
    짚었는지** 교차 확인한다. 벡터는 "의미가 비슷하다"만 보고 BM25는 "실제
    단어가 겹친다"만 보는, 서로 다른 종류의 신호라서 둘 다 동의해야 "진짜
    근거"로 인정 -- 벡터 혼자만의 우연한 근접성(오늘 사례들의 패턴)은 이제
    안 통과함.
    """
    relevant_docs = _filter_relevant_docs(similar_docs, bm25_origin_refs)
    if not relevant_docs:
        return "일반 지식(LLM 사전학습) 기반"
    counts: dict[str, int] = {}
    for d in relevant_docs:
        source = d.metadata.get("source") or "personal"
        counts[source] = counts.get(source, 0) + 1
    return " · ".join(f"{_SOURCE_LABELS.get(s, s)} {n}건" for s, n in counts.items()) + " 참고"


def _filter_relevant_docs(similar_docs: list, bm25_origin_refs: set | None = None) -> list:
    """SOURCE_RELEVANCE_FLOOR 이상"이면서 "BM25도 독립적으로 같은 문서를
    짚었을 때"만 "진짜 근거"로 인정(2026-10-14, 벡터 유사도 단독 기준으로는
    우연한 의미적 근접성을 못 걸러낸다는 게 실사용 중 재확인되어 보강).

    `bm25_origin_refs`가 None이면(호출부가 BM25 결과를 안 넘긴 경우) 이
    두 번째 조건은 건너뛴다 -- 기존처럼 유사도 크기만 본다(하위 호환).

    `_summarize_tier2_sources`(출처 표시용)와 `tier2_deep_agent`(일반지식
    옵션 끌 때 생성 여부 판단용) 둘 다 "근거가 있는가"를 똑같은 기준으로
    판단해야 해서 분리(2026-10-12, allow_general_knowledge 옵션 추가하며
    중복 로직 정리).
    """
    docs = [d for d in similar_docs if (d.metadata.get("similarity") or 0) >= SOURCE_RELEVANCE_FLOOR]
    if bm25_origin_refs is None:
        return docs
    return [d for d in docs if d.metadata.get("origin_ref") in bm25_origin_refs]


def tier2_deep_agent(state: NoteMateState) -> dict:
    """Tier2: 확신도 낮거나 명시적 질문일 때만 실행되는 딥 에이전트.

    - 장기 기억(note_chunks) 검색: 벡터 유사도 + BM25(Kiwi 형태소분석) 하이브리드
      검색(EnsembleRetriever, CLAUDE.md 기술 스펙 [확정])으로 top-k(5개) 필기를
      가져와 LLM에 컨텍스트로 제공. 벡터 검색만으론 "의미는 비슷한데 정확한
      용어가 다른" 경우를 놓칠 수 있어서, 키워드 기반 BM25로 보완한다.
    - LLM이 그 근거를 바탕으로 힌트(tier2_result)를 생성 -- 이게 "앵커 카드 보강" 내용.
    - `tier2_sources`: 그 힌트가 뭘 근거로 만들어졌는지(내 필기/강의자료/일반
      지식) 요약 문구 -- 2026-10-09, 완전히 새로운 내용일 때 검색 결과가
      없으면 LLM이 자기 사전학습 지식으로만 답하는 게(의도된 동작, 시스템
      프롬프트 참고) 사용자 입장에서는 "그냥 챗봇과 뭐가 다르냐"로 보일 수
      있다는 지적 + 강사 피드백("어떤 소스 기반인지 보여주는 게 중요한 부분")
      반영.
    - 웹검색(Tavily, 2026-10-12): "바로 물어보기" + 근거 없음일 때만 개입,
      조용한 힌트 경로는 절대 안 씀. 아래 _web_search_tavily() 참고.

    TODO:
      - 단기 기억(세션 내 InMemory VectorStore) 검색 -- 아직 세션 개념 없음
      - 원본 드릴다운(origin_ref로 원본 파일 실제 열람) -- 지금은 origin_ref 텍스트만 참고

    2026-10-15: 검색(DB)/생성(LLM) 호출을 전부 try/except로 감싼다 -- 외부 리뷰
    (노트메이트_리뷰종합_ClaudeGemini_2026-10-14.md Q3-b) 지적 + 직접 코드 확인으로
    발견한 실제 공백: 여태까지 이 함수엔 예외 처리가 전혀 없어서, 57일 전체를
    순차 재생하는 4차 사이클 도중 OpenAI API가 단 한 번이라도 타임아웃/일시
    오류를 내면 재생 스크립트 전체가 죽고 그 날의 나머지 필기가 통째로
    유실될 위험이 있었음. is_question을 try 블록 밖에서 먼저 계산해둬서,
    실패해도 "질문이면 침묵 대신 사유 문구"라는 기존 원칙(2026-10-14, 질문
    우선순위 설계와 같은 결)을 그대로 지킨다. 실패 사실은 log_failed_ingestion()
    (기존 Dead Letter 패턴, parse_vision_pdf_notes에서 쓰던 것과 동일)으로
    남겨서 조용히 삼키지 않는다.
    """
    is_question = bool((state.get("assessment") or {}).get("is_question"))
    try:
        return _tier2_deep_agent_inner(state, is_question)
    except Exception as e:  # noqa: BLE001 -- 외부 API(임베딩/검색/LLM/웹검색) 호출이
        # 일시적으로 실패해도 재생 스크립트 전체가 죽으면 안 되므로 여기서 막는다.
        log_failed_ingestion(
            origin_ref=state.get("origin_ref", "unknown"),
            error_type=type(e).__name__,
            error_message=str(e),
        )
        if is_question:
            return {
                "tier2_result": "일시적인 오류로 지금은 답변을 만들지 못했어요. "
                "잠시 후 다시 물어봐 주세요.",
                "tier2_sources": "오류로 힌트 생성 실패",
            }
        return {"tier2_result": None, "tier2_sources": "오류로 힌트 생성 실패"}


def _tier2_deep_agent_inner(state: NoteMateState, is_question: bool) -> dict:
    """tier2_deep_agent()의 실제 로직 -- 예외 처리는 바깥 함수가 전담."""
    search_course_ids = _resolve_search_course_ids(state)
    retriever, bm25 = build_hybrid_retriever(
        course_id=search_course_ids, k=5, query_embedding=state.get("embedding")
    )
    similar_docs = retriever.invoke(state["content"]) if retriever else []
    # BM25를 독립적으로 한 번 더 불러서(인덱스는 위에서 이미 만들어진 걸
    # 재사용 -- invoke() 자체는 가벼운 점수 계산만이라 비용 추가 미미)
    # "벡터가 찾은 문서를 BM25도 동의하는지" 교차 확인용 origin_ref 집합을
    # 만든다(2026-10-14, _filter_relevant_docs 설명 참고).
    bm25_docs = bm25.invoke(state["content"]) if bm25 else []
    bm25_origin_refs = {d.metadata.get("origin_ref") for d in bm25_docs}

    # 힌트 피드백 +/- 반영(2026-10-17, 2026-10-17 버그 수정: 질문 제외 필터와
    # 분리). 처음엔 위에서 이미 구한 similar_docs(근거 표시용, is_question
    # 제외 필터 적용됨)에서 그대로 뽑았는데, 실사용 중 발견: "바로 물어보기"/
    # 명시적 질문 줄은 전부 is_question=True라서 그 줄에 준 피드백이 이
    # 필터에 걸려 영원히 안 보임 -- 같은 질문에 `-`를 반복해서 줘도 전혀
    # 반영 안 되는 버그로 실측 재현됨(히말라야 산맥 질문 3연속 테스트).
    # 그래서 피드백 확인은 질문 제외 없이 별도로 검색한다(근거 표시 목적의
    # similar_docs/bm25_docs는 그대로 둠 -- 그쪽 필터는 여전히 유효한 목적이 있음).
    feedback_candidates = (
        search_similar_chunks(
            state.get("embedding"), search_course_ids, k=5, exclude_questions=False
        )
        if state.get("embedding")
        else []
    )
    negative_feedback_docs = [d for d in feedback_candidates if d.get("feedback_value") == "-"]
    positive_feedback_docs = [d for d in feedback_candidates if d.get("feedback_value") == "+"]
    if len(negative_feedback_docs) >= NEGATIVE_FEEDBACK_SUPPRESS_THRESHOLD and not is_question:
        return {
            "tier2_result": None,
            "tier2_sources": "반복된 부정적 피드백(-)으로 힌트 생성 보류",
        }

    # allow_general_knowledge=False(2026-10-12 추가) + 근거 자료가 하나도 없으면
    # LLM 호출 자체를 건너뛴다 -- "검색은 됐는데 생성 내용이 부정확"한 것과는
    # 다른 문제(근거 자체가 아예 없는 경우)이지만, 둘 다 "힌트 남발" 완화에
    # 기여하고 비교테스트에도 쓸 수 있어 추가(사용자 요청). 기본값 True면
    # 기존 동작(일반 지식으로라도 답변) 그대로 유지.
    #
    # is_ask(바로 물어보기)는 여기서 제외한다 -- 사용자 정정(2026-10-12,
    # 아래 웹검색 분기 설명 참고): 조용한 힌트는 완전히 침묵해도 되지만,
    # 바로 물어보기는 사용자가 직접 물은 "질문"이라 무응답으로 넘기면 안 됨.
    # 대신 바로 아래 웹검색 분기가 먼저 기회를 갖고, 거기서도 못 찾으면
    # 그쪽이 "못 찾았다"는 정직한 답변을 책임진다.
    #
    # 2026-10-14 보강(사용자 지적): is_ask가 아니어도(필기창에 그냥 물음표로
    # 적은 질문도) "질문"이면 똑같이 완전 침묵은 안 됨 -- "질문은 사용자가
    # 명시적으로 알고 싶다는 의도 표시"라서, 답을 못 내놓는 경우에도 "왜
    # 못 내놓는지"가 답변 내용에 명시적으로 나와야 한다는 원칙(위 라우팅
    # 우선순위 변경과 같은 결). 이 조합(질문 + 근거 없음 + 일반지식 옵션
    # 꺼짐 + is_ask 아님)은 바로 아래 웹검색 분기도 안 타므로(is_ask 전용),
    # 여기서 바로 그 이유를 명시한 정형 문구로 답한다 -- LLM이 옵션을
    # 무시하고 몰래 일반지식으로 답해버리면 안 되므로 LLM 호출 없이 결정론적으로.
    # (is_question은 이제 바깥 tier2_deep_agent()가 계산해서 넘겨준다 -- 2026-10-15)
    if (
        not state.get("allow_general_knowledge", True)
        and not state.get("is_ask")
        and not _filter_relevant_docs(similar_docs, bm25_origin_refs)
    ):
        if is_question:
            return {
                "tier2_result": "강의자료·내 필기 어디에서도 관련 정보를 찾지 못했고, "
                "LLM 기본지식 옵션도 꺼져 있어서 답변을 만들지 못했어요. "
                "옵션을 켜거나 질문을 조금 더 구체적으로 다시 적어주세요.",
                "tier2_sources": "근거 자료 없음 (일반지식 옵션 꺼짐)",
            }
        return {
            "tier2_result": None,
            "tier2_sources": "근거 자료 없음 -- 일반지식 옵션 꺼짐으로 힌트 생성 건너뜀",
        }

    # 웹검색(Tavily, 2026-10-12) -- "바로 물어보기"(is_ask) + 내 필기/강의자료에
    # 근거가 없을 때만 개입. 조용한 힌트 경로(is_ask=False)는 절대 안 씀
    # ([[deferred-web-search-addon]] 스코프: 바로 물어보기 전용). 이미 근거가
    # 충분하면(검색이 잘 됐으면) 웹검색은 호출 자체를 안 해서 속도 영향도 없음.
    web_results = []
    if state.get("is_ask") and not _filter_relevant_docs(similar_docs, bm25_origin_refs):
        web_results = _web_search_tavily(state["content"])
        if not web_results:
            # 바로 물어보기 + 내 필기/강의자료 근거 없음 + 웹검색도 결과 없음(또는
            # 실패/타임아웃) -- 사용자 정정(2026-10-12): "이 경우엔 답변이지" --
            # 바로 물어보기는 사용자가 직접 물어본 "질문"이라 조용한 힌트처럼
            # 완전히 무응답으로 넘기면 안 됨(라우팅 원칙 "질문엔 무조건 답한다"와
            # 같은 결). 대신 LLM 혼자 추측하게 두지 않고(이 기능 전체의 목적),
            # 솔직하게 "못 찾았다"고 답하는 정형 문구로 응답 -- LLM 호출 없이
            # 결정론적으로 생성(추측 여지 자체를 차단). 조용한 힌트 경로
            # (is_ask=False)는 이 분기를 안 타므로 기존 "일반 지식" 폴백 그대로 유지.
            return {
                "tier2_result": "강의자료·내 필기·웹검색 어디에서도 관련 정보를 찾지 못했어요. "
                "질문을 조금 더 구체적으로 다시 적어주시면 다시 찾아볼게요.",
                "tier2_sources": "근거 자료 없음 (웹검색도 결과 없음)",
            }

    context_lines = [f"- ({d.metadata.get('source')}) {d.page_content}" for d in similar_docs]
    context_lines += [f"- (웹검색: {r.get('title')}) {r.get('content')}" for r in web_results]
    context = "\n".join(context_lines) if context_lines else "(참고할 만한 이전 필기 없음)"

    assessment = state.get("assessment") or {}
    # 웹검색이 실제로 개입한 경우(= 로컬 자료에 근거 없음 + 바로 물어보기)
    # 에만 적용되는 추가 지시(2026-10-14, 사용자 요청): "LLM 기본지식만으로
    # 추측하지 말고, 웹검색 결과와 비교해서 대체로 일치하면 그대로 답하고,
    # 불일치하면 웹검색 쪽이 더 최신 정보일 가능성이 높으니 그걸 우선해서
    # 종합하라"는 명시적 비교 지시를 추가. 이전엔 웹검색 결과를 그냥 컨텍스트
    # 목록에 끼워 넣기만 하고 "어떻게 활용할지"는 LLM 재량에 맡겼는데, 그러면
    # LLM이 자기 사전학습 지식을 그대로 내보내고 웹검색 결과는 사실상 무시할
    # 위험이 있음(베타테스트 중 관찰된 패턴).
    compare_instruction = ""
    if web_results:
        compare_instruction = (
            "\n\n위 참고자료에는 웹검색 결과가 포함되어 있습니다. 당신이 사전학습으로 "
            "이미 알고 있는 지식과 이 웹검색 결과를 비교하세요 -- 대체로 일치하면 "
            "그 내용 그대로 답변하고, 만약 서로 다르다면 웹검색 결과가 더 최신 정보일 "
            "가능성이 높으니 그것을 우선해서 반영한 답변을 작성하세요."
        )
    # 힌트 피드백 +/- 반영(2026-10-17) -- 위에서 억제 임계치는 안 넘었지만
    # `-`가 하나라도 있으면 "다르게 설명"을, 없고 `+`가 있으면 "이 방식 유지"를
    # 지시한다. 둘 다 있을 수는 없음(있으면 `-`를 우선 -- 개선 시도가 더
    # 급한 신호).
    feedback_instruction = ""
    if negative_feedback_docs:
        past_attempt = negative_feedback_docs[0].get("tier2_result") or "(내용 없음)"
        if is_question or state.get("is_ask"):
            # 질문/바로 물어보기에 대한 `-`는 "답하지 말라"가 아니라 "제대로
            # 답하라"는 뜻(2026-10-17, 사용자 지적) -- 단순 문체 변경이 아니라
            # 정확성/점검 쪽에 무게를 둔 지시로 구분. 억제 분기(위)가 이미
            # `not is_question`일 때만 동작하도록 가드돼 있어서, 질문은 여기
            # 오는 경우에도 완전 침묵은 안 되고 항상 답이 나간다.
            feedback_instruction = (
                f"\n\n참고: 비슷한 질문에 대해 이전에 다음과 같이 답했으나 도움이 "
                f'안 됐다는 피드백이 있었습니다: "{past_attempt}" -- 이번에는 내용을 '
                "다시 한 번 점검해서, 빠지거나 부정확한 부분이 없는지 확인하고 "
                "더 정확하고 구체적으로 답변하세요(단순히 표현만 바꾸지 말 것)."
            )
        else:
            feedback_instruction = (
                f"\n\n참고: 비슷한 내용에 대해 이전에 다음과 같이 설명했으나 도움이 "
                f'안 됐다는 피드백이 있었습니다: "{past_attempt}" -- 이번에는 다른 '
                "방식(다른 비유, 더 구체적인 예시, 더 짧거나 더 자세하게 등)으로 "
                "설명해보세요."
            )
    elif positive_feedback_docs:
        past_attempt = positive_feedback_docs[0].get("tier2_result") or "(내용 없음)"
        feedback_instruction = (
            f"\n\n참고: 비슷한 내용에 대해 이전에 다음과 같이 설명했더니 도움이 "
            f'됐다는 피드백이 있었습니다: "{past_attempt}" -- 이번에도 비슷한 '
            "방식을 유지하세요."
        )
    user_prompt = (
        f"방금 필기: {state['content']}\n\n"
        f"판단: is_question={assessment.get('is_question')}, "
        f"is_abbreviated={assessment.get('is_abbreviated')}, "
        f"is_new_concept={assessment.get('is_new_concept')}\n\n"
        f"참고할 이전 필기/웹검색 결과 (관련도 순):\n{context}"
        f"{compare_instruction}"
        f"{feedback_instruction}"
    )

    response = invoke_chat(
        [
            {"role": "system", "content": _TIER2_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        purpose="tier2_hint",
    )
    # 웹검색이 실제로 개입했을 때만 출처 표시에 반영 -- 위 호출 조건상 web_results가
    # 채워지는 경우는 항상 _filter_relevant_docs(similar_docs)가 비어있을 때라서
    # (= _summarize_tier2_sources가 "일반 지식" 라벨을 돌려줄 상황), 그 라벨을
    # "웹검색 N건 참고"로 바꿔치기하면 된다.
    sources = (
        f"웹검색 {len(web_results)}건 참고"
        if web_results
        else _summarize_tier2_sources(similar_docs, bm25_origin_refs)
    )
    return {"tier2_result": response.content, "tier2_sources": sources}


def _web_search_tavily(query: str, max_results: int = 3, timeout: float = 5.0) -> list[dict]:
    """"바로 물어보기" 전용 웹검색(Tavily) -- 내 필기/강의자료에 근거가 없을 때만
    tier2_deep_agent가 호출한다(위 호출부 참고). API 실패/타임아웃/키 없음은
    전부 조용히 빈 리스트로 처리 -- 바로 물어보기 응답 전체가 외부 API 장애로
    멈추면 안 되므로(2026-10-12, 발표 시연 안전장치). 라우팅/다른 로직엔 영향
    없음 -- 실패해도 그냥 "근거 없음" 상태로 평소처럼 진행됨.
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return []
    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"query": query, "max_results": max_results},
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json().get("results", [])
    except Exception:
        return []


def store_only(state: NoteMateState) -> dict:
    """Tier1까지만 (조용히 저장만, 화면에 표시 안 함).

    실제 DB 저장은 이 노드가 아니라 공통 persist_note 노드(app.py에서 이 다음에
    연결됨)가 담당한다 -- Tier1/2 상관없이 저장 로직은 하나만 있으면 되므로.
    이 노드는 "Tier2 개입 없이 지나갔다"는 그래프 분기 지점 역할만 한다.
    """
    return {}


def _classify_hint_category(assessment: dict, has_hint: bool) -> str | None:
    """힌트 유형을 추가 LLM 호출 없이, 이미 계산된 assessment 신호로 결정론적으로
    분류한다(2026-10-08, 힌트 피드백 Phase 4 선행 작업 -- "유사 유형/주제/형식별
    피드백 합계"를 나중에 묶을 축). 힌트가 없는 줄(store_only)은 분류 의미가
    없어 None. 우선순위는 compute_routing_decision의 분기 순서와 동일하게 맞춤
    (is_question > is_new_concept > is_abbreviated > 그 외).
    """
    if not has_hint:
        return None
    if assessment.get("is_question"):
        return "question_answer"
    if assessment.get("is_new_concept"):
        return "new_concept"
    if assessment.get("is_abbreviated"):
        return "abbreviated_reminder"
    return "general_nudge"


def persist_note(state: NoteMateState) -> dict:
    """Tier1/2 경로 상관없이 공통으로 실행 -- 판단 결과를 note_chunks에 저장.

    가이드 원문은 "세션 종료/유휴 시점 배치 upsert"라고 돼 있지만, 지금 단계는
    재생 스크립트로 한 건씩 순차 테스트하는 거라 즉시 저장으로 단순화했다.
    [잠정] 나중에 실제 세션 개념이 생기면 배치 방식으로 바꿀 수 있음.

    dry_run이면 여기서 끝 -- DB insert도, 세션 작업기억 등록도 안 한다. 그
    앞 노드들(assess_note/tier2_deep_agent 등)은 전부 읽기만 하므로, 검색/
    힌트 생성은 기존 데이터 기준으로 실제로 돌아간 상태 그대로 state에 남아
    호출자에게 반환된다(2026-10-10, test:: 테스트 입력이 RAG를 오염시키지
    않으면서 힌트 품질을 확인할 수 있게).
    """
    if state.get("dry_run"):
        return {}

    assessment = state.get("assessment") or {}
    row = {
        "content": state["content"],
        "source": state["source"],
        "medium": state["medium"],
        "course_id": state.get("course_id"),
        "note_timestamp": state.get("note_timestamp"),
        "origin_ref": state["origin_ref"],
        "top_score": state.get("top_score"),
        "is_question": assessment.get("is_question"),
        "is_abbreviated": assessment.get("is_abbreviated"),
        "is_new_concept": assessment.get("is_new_concept"),
        "session_duplicate_score": state.get("session_duplicate_score"),
        "routing_decision": state.get("routing_decision"),
        "tier2_result": state.get("tier2_result"),
        "embedding": state.get("embedding"),
        "client_line_id": state.get("client_line_id"),
        "hint_category": _classify_hint_category(assessment, bool(state.get("tier2_result"))),
        "top_match_feedback_value": state.get("top_match_feedback_value"),
        "top_match_feedback_days_ago": state.get("top_match_feedback_days_ago"),
        # 2026-10-08: session_id는 지금까지 세션 작업기억(중복 힌트 방지용,
        # 인메모리)에만 쓰이고 note_chunks엔 저장 안 됐음 -- 그래서
        # 새로고침하면 같은 날 여러 세션 경계가 화면에서 다시 사라지는
        # 문제가 있었음(사용자 지적). 프론트엔드가 세션 구분선을
        # 재구성하는 데 쓸 수 있게 같이 저장한다.
        "session_id": state.get("session_id"),
        # 2026-10-09: 힌트가 뭘 근거로 만들어졌는지(내 필기/강의자료/일반
        # 지식) 요약 -- tier2_deep_agent._summarize_tier2_sources() 참고.
        # 힌트가 없는 줄(store_only)은 당연히 None.
        "tier2_sources": state.get("tier2_sources"),
        # 2026-10-10: 과거 세션/날짜 블록에 끼워넣은 입력인지 표시용 플래그
        # ([[deferred-historical-note-insertion]]). 판단/라우팅엔 안 쓰임.
        "inserted_out_of_place": state.get("inserted_out_of_place"),
        # 2026-10-12: 자료 업로드 UI의 분류명 드롭다운에서 고르거나 새로
        # 만든 값. 판단/라우팅엔 안 쓰이고 그대로 저장만 됨.
        "upload_label": state.get("upload_label"),
    }
    insert_note_chunk(row)

    # 세션 작업기억에도 등록 -- 다음 필기가 "이거 방금 다뤘잖아" 판단할 수 있게.
    # session_id가 없으면(세션 개념 안 쓰는 호출) 조용히 건너뜀.
    session_id = state.get("session_id")
    if session_id and state.get("embedding"):
        add_to_session(session_id, state["embedding"])

    return {}

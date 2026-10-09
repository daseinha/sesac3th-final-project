"""LangGraph 노드 함수 모음.

assess_note는 실제 로직(임베딩 + 벡터 검색 + LLM 구조화 출력)으로 구현됐다.
tier2_deep_agent, store_only는 아직 스텁 -- 각 함수의 TODO 참고.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from src.db import insert_note_chunk, search_top_similarity
from src.llm import embed_text, invoke_chat, invoke_structured
from src.retrieval import build_hybrid_retriever
from src.routers import LOW_THRESHOLD, compute_routing_decision
from src.session_memory import add_to_session, check_session_duplicate
from src.state import NoteAssessment, NoteMateState


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

    top_match = search_top_similarity(embedding, course_id=state.get("course_id"))
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


def _summarize_tier2_sources(similar_docs: list) -> str:
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
    있으니(`metadata['similarity']`), 라우팅에서 이미 쓰는 `LOW_THRESHOLD`
    (새 임의의 숫자를 만들지 않고 기존 검증된 값 재사용)보다 낮은 매치는
    "근거"로 안 센다. BM25 전용 매치(벡터 점수가 없는 것, `similarity is None`)
    는 신뢰할 점수가 없어서 같이 제외 -- 이번 실측에서 바로 이런 매치들이
    키워드만 우연히 겹친 가짜 근거였음.
    """
    relevant_docs = [d for d in similar_docs if (d.metadata.get("similarity") or 0) >= LOW_THRESHOLD]
    if not relevant_docs:
        return "일반 지식(LLM 사전학습) 기반"
    counts: dict[str, int] = {}
    for d in relevant_docs:
        source = d.metadata.get("source") or "personal"
        counts[source] = counts.get(source, 0) + 1
    return " · ".join(f"{_SOURCE_LABELS.get(s, s)} {n}건" for s, n in counts.items())


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

    TODO:
      - 단기 기억(세션 내 InMemory VectorStore) 검색 -- 아직 세션 개념 없음
      - 원본 드릴다운(origin_ref로 원본 파일 실제 열람) -- 지금은 origin_ref 텍스트만 참고
    """
    retriever = build_hybrid_retriever(
        course_id=state.get("course_id"), k=5, query_embedding=state.get("embedding")
    )
    similar_docs = retriever.invoke(state["content"]) if retriever else []

    if similar_docs:
        context = "\n".join(f"- ({d.metadata.get('source')}) {d.page_content}" for d in similar_docs)
    else:
        context = "(참고할 만한 이전 필기 없음)"

    assessment = state.get("assessment") or {}
    user_prompt = (
        f"방금 필기: {state['content']}\n\n"
        f"판단: is_question={assessment.get('is_question')}, "
        f"is_abbreviated={assessment.get('is_abbreviated')}, "
        f"is_new_concept={assessment.get('is_new_concept')}\n\n"
        f"참고할 이전 필기 (유사도 순):\n{context}"
    )

    response = invoke_chat(
        [
            {"role": "system", "content": _TIER2_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        purpose="tier2_hint",
    )
    return {"tier2_result": response.content, "tier2_sources": _summarize_tier2_sources(similar_docs)}


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
    """
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
    }
    insert_note_chunk(row)

    # 세션 작업기억에도 등록 -- 다음 필기가 "이거 방금 다뤘잖아" 판단할 수 있게.
    # session_id가 없으면(세션 개념 안 쓰는 호출) 조용히 건너뜀.
    session_id = state.get("session_id")
    if session_id and state.get("embedding"):
        add_to_session(session_id, state["embedding"])

    return {}

"""LangGraph 노드 함수 모음.

assess_note는 실제 로직(임베딩 + 벡터 검색 + LLM 구조화 출력)으로 구현됐다.
tier2_deep_agent, store_only는 아직 스텁 -- 각 함수의 TODO 참고.
"""

from pydantic import BaseModel, Field

from src.db import insert_note_chunk, search_top_similarity
from src.llm import embed_text, invoke_chat, invoke_structured
from src.retrieval import build_hybrid_retriever
from src.routers import compute_routing_decision
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


_ASSESSMENT_SYSTEM_PROMPT = (
    "당신은 학습자의 실시간 필기를 분석하는 어시스턴트입니다. "
    "주어진 필기 한 조각을 보고 아래 세 가지를 판단하세요.\n"
    "1. is_question: 명시적인 질문을 담고 있는가\n"
    "2. is_abbreviated: 축약이 심해 나중에 봤을 때 이해하기 어려운가\n"
    "3. is_new_concept: 새로운 개념을 다루는가\n"
    "필기 원문 외의 맥락은 알 수 없으니, 주어진 텍스트만 보고 판단하세요."
)


def assess_note(state: NoteMateState) -> dict:
    """정리 노드: 신호 A(top_score) + 신호 B(assessment) 계산.

    - 신호 A: 임베딩 생성 -> note_chunks에서 코사인 유사도 검색.
      비교할 기존 데이터가 없으면(첫 필기 등) top_score=0.0으로 처리한다
      ("지원 자료 없음"은 확신도가 낮은 것과 동일하게 취급 -> 자연히 tier2로 라우팅됨).
    - 신호 B: LLM 구조화 출력으로 필기 완결도 판단.

    TODO: 세션 내 작업기억(InMemory VectorStore) 중복 검사 -> session_duplicate_score
    """
    content = state["content"]

    embedding = embed_text(content, purpose="assess_note_embedding")

    raw_top_score = search_top_similarity(embedding, course_id=state.get("course_id"))
    top_score = raw_top_score if raw_top_score is not None else 0.0

    result: _NoteAssessmentSchema = invoke_structured(
        _NoteAssessmentSchema,
        [
            {"role": "system", "content": _ASSESSMENT_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        purpose="assess_note_assessment",
    )
    assessment: NoteAssessment = {
        "is_question": result.is_question,
        "is_abbreviated": result.is_abbreviated,
        "is_new_concept": result.is_new_concept,
    }

    return {
        "embedding": embedding,
        "top_score": top_score,
        "assessment": assessment,
        "session_duplicate_score": None,  # TODO: 세션 내 작업기억 구현 후 채움
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


def tier2_deep_agent(state: NoteMateState) -> dict:
    """Tier2: 확신도 낮거나 명시적 질문일 때만 실행되는 딥 에이전트.

    - 장기 기억(note_chunks) 검색: 벡터 유사도 + BM25(Kiwi 형태소분석) 하이브리드
      검색(EnsembleRetriever, CLAUDE.md 기술 스펙 [확정])으로 top-k(5개) 필기를
      가져와 LLM에 컨텍스트로 제공. 벡터 검색만으론 "의미는 비슷한데 정확한
      용어가 다른" 경우를 놓칠 수 있어서, 키워드 기반 BM25로 보완한다.
    - LLM이 그 근거를 바탕으로 힌트(tier2_result)를 생성 -- 이게 "앵커 카드 보강" 내용.

    TODO:
      - 단기 기억(세션 내 InMemory VectorStore) 검색 -- 아직 세션 개념 없음
      - 원본 드릴다운(origin_ref로 원본 파일 실제 열람) -- 지금은 origin_ref 텍스트만 참고
    """
    retriever = build_hybrid_retriever(course_id=state.get("course_id"), k=5)
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
    return {"tier2_result": response.content}


def store_only(state: NoteMateState) -> dict:
    """Tier1까지만 (조용히 저장만, 화면에 표시 안 함).

    실제 DB 저장은 이 노드가 아니라 공통 persist_note 노드(app.py에서 이 다음에
    연결됨)가 담당한다 -- Tier1/2 상관없이 저장 로직은 하나만 있으면 되므로.
    이 노드는 "Tier2 개입 없이 지나갔다"는 그래프 분기 지점 역할만 한다.
    """
    return {}


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
        "embedding": state.get("embedding"),
    }
    insert_note_chunk(row)
    return {}

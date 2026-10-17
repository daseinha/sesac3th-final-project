"""하이브리드 검색 -- 벡터 유사도 + BM25(Kiwi 형태소분석) 결합.

CLAUDE.md 기술 스펙 [확정]: "벡터 유사도 + BM25/Kiwi 형태소분석 (EnsembleRetriever)".
Tier2 딥 에이전트가 참고 필기를 찾을 때 씀. assess_note의 top_score(신호 A)는
스펙에 명시된 대로 "벡터 검색시 나오는 유사도 점수" 그대로 순수 벡터만 쓴다 --
이건 확신도 신호라 단순하고 저렴한 게 원래 의도라, 여기서는 안 바꿈.
"""

from kiwipiepy import Kiwi
from langchain_classic.retrievers import EnsembleRetriever
from langchain_community.retrievers.bm25 import BM25Retriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.db import fetch_corpus_for_bm25, search_similar_chunks
from src.llm import embed_text

_kiwi = Kiwi()


def kiwi_tokenize(text: str) -> list[str]:
    """BM25Retriever의 preprocess_func으로 쓰는 Kiwi 형태소 토크나이저.

    공백 기준으로만 쪼개면 한국어 조사/어미 때문에 같은 단어도 다르게 매칭되니
    (예: "그래프는" vs "그래프가"), 형태소 단위로 쪼개서 어근을 맞춘다.
    """
    return [token.form for token in _kiwi.tokenize(text)]


class _VectorRetriever(BaseRetriever):
    """note_chunks 벡터 유사도 검색을 LangChain Retriever 인터페이스로 감싼 것.

    EnsembleRetriever가 여러 Retriever를 같은 인터페이스로 다뤄야 해서 필요함.
    """

    # 2026-10-17 야간: 노트 묶음(RAG 공유그룹) -- 리스트를 받으면 묶인 노트
    # 전체를 검색 범위로 삼는다(search_similar_chunks로 그대로 전달).
    course_id: str | list[str] | None = None
    k: int = 5
    # 호출부(tier2_deep_agent)가 assess_note에서 이미 계산해둔 임베딩을 넘겨주면
    # 그걸 그대로 쓴다 -- 같은 필기 내용을 또 임베딩하는 중복 호출을 피하기
    # 위함(2026-10-02 실측: Tier2 건마다 임베딩 호출이 2배로 나가던 것 발견).
    # 안 넘겨주면(다른 호출부가 생길 경우 대비) 기존처럼 쿼리에서 새로 계산.
    embedding: list[float] | None = None

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        embedding = self.embedding or embed_text(query, purpose="hybrid_search_query_embedding")
        chunks = search_similar_chunks(embedding, course_id=self.course_id, k=self.k)
        return [
            Document(
                page_content=c["content"],
                metadata={
                    "source": c["source"],
                    "origin_ref": c.get("origin_ref"),
                    "similarity": c["similarity"],
                    # 힌트 피드백 반영 로직(2026-10-17)이 쓰는 두 필드 --
                    # tier2_deep_agent 참고.
                    "feedback_value": c.get("feedback_value"),
                    "tier2_result": c.get("tier2_result"),
                },
            )
            for c in chunks
        ]


def build_hybrid_retriever(
    course_id: str | list[str] | None, k: int = 5, query_embedding: list[float] | None = None
) -> tuple[EnsembleRetriever | None, BM25Retriever | None]:
    """벡터 검색 + BM25(Kiwi) 결합 리트리버 생성.

    BM25 인덱스는 course_id 범위의 note_chunks를 매번 새로 불러와 즉석에서
    만든다(LangChain BM25Retriever가 인메모리 방식이라서 -- MVP 단계에서는
    코퍼스가 수천 건 수준이라 감당 가능, 나중에 규모 커지면 캐싱 고려).
    코퍼스가 비어있으면 (None, None) -- 호출부에서 벡터 검색만으로 대체 처리.

    query_embedding: 호출부가 쿼리 텍스트의 임베딩을 이미 갖고 있으면 넘겨서
    벡터 검색 쪽이 또 임베딩하지 않게 한다(2026-10-02, _VectorRetriever 참고).

    반환값이 (ensemble, bm25) 튜플로 바뀜(2026-10-14) -- "힌트 근거 표시"가
    벡터 유사도만 보고 판단하다가(순수 의미 유사도만으론 "AI", "모델" 같은
    흔한 단어가 겹치는 완전히 무관한 질문도 통과시켜버리는 오탐이 실사용 중
    확인됨), BM25도 같은 문서를 짚었는지 교차 확인하도록 바꾸기 위함
    (src/nodes.py의 _filter_relevant_docs 참고). bm25 자체를 따로 노출해두면
    호출부가 코퍼스를 또 가져와 인덱스를 다시 만드는 비용(여기가 진짜 느린
    부분) 없이 `bm25.invoke()`(가벼운 점수 계산만)만 한 번 더 부르면 됨.
    """
    corpus = fetch_corpus_for_bm25(course_id)
    if not corpus:
        return None, None

    texts = [row["content"] for row in corpus]
    metadatas = [
        {
            "source": row["source"],
            "origin_ref": row["origin_ref"],
            # 힌트 피드백 반영 로직(2026-10-17) -- 벡터 경로와 동일하게 BM25
            # 매치에도 실어서, 어느 경로로 찾았든 과거 +/- 피드백을 알 수 있게.
            "feedback_value": row.get("feedback_value"),
            "tier2_result": row.get("tier2_result"),
        }
        for row in corpus
    ]
    bm25 = BM25Retriever.from_texts(texts, metadatas=metadatas, preprocess_func=kiwi_tokenize)
    bm25.k = k

    vector_retriever = _VectorRetriever(course_id=course_id, k=k, embedding=query_embedding)

    ensemble = EnsembleRetriever(retrievers=[vector_retriever, bm25], weights=[0.5, 0.5])
    return ensemble, bm25

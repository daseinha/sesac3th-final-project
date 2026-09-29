"""LLM / 임베딩 클라이언트.

CLAUDE.md 기술 스펙의 "provider를 코드에 하드코딩하지 말 것" 원칙에 따라,
채팅 모델은 langchain의 init_chat_model 패턴을 쓴다. .env의 LLM_MODEL 값만
바꾸면 provider가 통째로 바뀐다 (예: "gpt-4.1-mini" -> "claude-sonnet-5").
"""

import os

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import OpenAIEmbeddings

_LLM_MODEL = os.environ.get("LLM_MODEL", "gpt-4.1-mini")
_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")

_chat_model: BaseChatModel | None = None
_embeddings_model: OpenAIEmbeddings | None = None


def get_chat_model() -> BaseChatModel:
    """구조화 출력(판단 노드)에 쓰는 채팅 모델. 프로세스당 한 번만 생성해 재사용."""
    global _chat_model
    if _chat_model is None:
        _chat_model = init_chat_model(_LLM_MODEL, temperature=0)
    return _chat_model


def get_embeddings_model() -> OpenAIEmbeddings:
    """임베딩 모델. 프로세스당 한 번만 생성해 재사용.

    임베딩은 (CLAUDE.md 결정에 따라) OpenAI 계열로 고정 -- LLM처럼 provider를
    자유롭게 바꾸는 요구사항은 없었어서 OpenAIEmbeddings를 직접 씀.
    """
    global _embeddings_model
    if _embeddings_model is None:
        _embeddings_model = OpenAIEmbeddings(model=_EMBEDDING_MODEL)
    return _embeddings_model

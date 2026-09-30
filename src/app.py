from langgraph.graph import END, START, StateGraph

from src.nodes import (
    assess_note,
    decide_routing,
    embed_and_store,
    persist_note,
    store_only,
    tier2_deep_agent,
)
from src.routers import route, route_by_source
from src.state import NoteMateState

builder = StateGraph(NoteMateState)

builder.add_node("assess_note", assess_note)
builder.add_node("decide_routing", decide_routing)
builder.add_node("tier2_deep_agent", tier2_deep_agent)
builder.add_node("store_only", store_only)
builder.add_node("embed_and_store", embed_and_store)
builder.add_node("persist_note", persist_note)

# 첫 갈림길: source="personal"만 Tier1/2 전체 파이프라인, 나머지(업로드된
# 참고자료)는 임베딩만 만들고 바로 저장하는 가벼운 경로로 보낸다.
builder.add_conditional_edges(
    START,
    route_by_source,
    {"full_pipeline": "assess_note", "reference_only": "embed_and_store"},
)

builder.add_edge("assess_note", "decide_routing")
builder.add_conditional_edges(
    "decide_routing",
    route,
    {"tier2": "tier2_deep_agent", "store_only": "store_only"},
)
# Tier1/2 상관없이 판단 결과는 공통으로 저장 (persist_note)
builder.add_edge("tier2_deep_agent", "persist_note")
builder.add_edge("store_only", "persist_note")
builder.add_edge("embed_and_store", "persist_note")
builder.add_edge("persist_note", END)

graph = builder.compile()

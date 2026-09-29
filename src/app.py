from langgraph.graph import END, START, StateGraph

from src.nodes import (
    assess_note,
    decide_routing,
    persist_note,
    store_only,
    tier2_deep_agent,
)
from src.routers import route
from src.state import NoteMateState

builder = StateGraph(NoteMateState)

builder.add_node("assess_note", assess_note)
builder.add_node("decide_routing", decide_routing)
builder.add_node("tier2_deep_agent", tier2_deep_agent)
builder.add_node("store_only", store_only)
builder.add_node("persist_note", persist_note)

builder.add_edge(START, "assess_note")
builder.add_edge("assess_note", "decide_routing")
builder.add_conditional_edges(
    "decide_routing",
    route,
    {"tier2": "tier2_deep_agent", "store_only": "store_only"},
)
# Tier1/2 상관없이 판단 결과는 공통으로 저장 (persist_note)
builder.add_edge("tier2_deep_agent", "persist_note")
builder.add_edge("store_only", "persist_note")
builder.add_edge("persist_note", END)

graph = builder.compile()

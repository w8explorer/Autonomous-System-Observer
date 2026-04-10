"""
Routing functions for workflow conditional edges
"""

from models.state import GraphState


def route_after_quality_check(state: GraphState) -> str:
    """Route after query quality check"""
    if state.get("query_needs_improvement", False):
        return "rewrite_query"
    else:
        return "retrieve"


def route_after_grading(state: GraphState) -> str:
    """Route after document grading"""
    if state["any_relevant"]:
        return "generate"
    elif state.get("retry_count", 0) < 2:
        return "rewrite_query"
    else:
        return "generate"


def route_after_reflection(state: GraphState) -> str:
    """Route after self-reflection"""
    if state.get("answer_quality_good", False):
        return "evaluation"
    elif state.get("retry_count", 0) < 2:
        return "rewrite_query"
    else:
        return "evaluation"

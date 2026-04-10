"""
LangGraph workflow builder
"""

from langgraph.graph import StateGraph, END

from models.state import GraphState
from .nodes import WorkflowNodes
from .routing import route_after_quality_check, route_after_grading, route_after_reflection


def build_workflow(llm, retriever):
    """
    Build the LangGraph workflow for code intelligence

    Args:
        llm: Language model instance
        retriever: Vector store retriever

    Returns:
        Compiled workflow graph
    """
    # Initialize nodes
    nodes = WorkflowNodes(llm, retriever)

    # Build workflow
    workflow = StateGraph(GraphState)

    # Add all nodes
    workflow.add_node("query_quality", nodes.query_quality_node)
    workflow.add_node("retrieve", nodes.retrieve_node)
    workflow.add_node("grade_documents", nodes.grade_documents_node)
    workflow.add_node("rewrite_query", nodes.rewrite_query_node)
    workflow.add_node("generate", nodes.generate_node)
    workflow.add_node("self_reflection", nodes.self_reflection_node)
    workflow.add_node("evaluation", nodes.evaluation_node)
    workflow.add_node("finalize", nodes.finalize_node)

    # Set entry point - start with quality check
    workflow.set_entry_point("query_quality")

    # Add edges
    workflow.add_conditional_edges(
        "query_quality",
        route_after_quality_check,
        {"rewrite_query": "rewrite_query", "retrieve": "retrieve"}
    )
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        route_after_grading,
        {"generate": "generate", "rewrite_query": "rewrite_query"}
    )
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate", "self_reflection")
    workflow.add_conditional_edges(
        "self_reflection",
        route_after_reflection,
        {"rewrite_query": "rewrite_query", "evaluation": "evaluation"}
    )
    workflow.add_edge("evaluation", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()

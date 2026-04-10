"""
State models for the LangGraph workflow
"""

from typing import TypedDict, List, Dict


class GraphState(TypedDict):
    """State schema for the RAG workflow graph"""

    # Query-related fields
    question: str
    query_needs_improvement: bool
    rewritten_query: str

    # Retrieval-related fields
    retrieved_code: List[str]
    code_files: List[str]
    retrieved_documents: List  # Full Document objects for evaluators

    # Grading-related fields
    grading_scores: List[Dict[str, str]]
    any_relevant: bool

    # Generation-related fields
    generation: str
    answer_quality_good: bool
    final_answer: str

    # Evaluation-related fields
    evaluation_results: Dict

    # Process tracking
    intermediate_steps: List[str]
    retry_count: int

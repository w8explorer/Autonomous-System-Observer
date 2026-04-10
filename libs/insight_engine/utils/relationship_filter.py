"""
Relationship-based document filtering utilities
"""

from typing import List
from langchain_core.documents import Document

from config.settings import RELATIONSHIP_KEYWORDS
from .java_parser import extract_target_entity


def filter_by_relationships(documents: List[Document], question: str) -> List[Document]:
    """
    Filter retrieved documents based on relationship queries

    Args:
        documents: List of retrieved documents
        question: User's question

    Returns:
        Filtered list of documents based on relationships
    """
    question_lower = question.lower()

    # Check if it's a relationship query
    if not any(keyword in question_lower for keyword in RELATIONSHIP_KEYWORDS):
        return documents  # Not a relationship query, return all

    # Extract target entity from question
    target = extract_target_entity(question)
    if not target:
        return documents

    filtered = []
    for doc in documents:
        metadata = doc.metadata

        # Check different relationship types
        if 'calls' in question_lower or 'uses' in question_lower:
            # Check if this code calls the target
            if any(target in call for call in metadata.get('method_calls', [])):
                filtered.append(doc)
        elif 'depends' in question_lower or 'imports' in question_lower:
            # Check if this code imports the target
            if any(target in imp for imp in metadata.get('imports', [])):
                filtered.append(doc)
        elif 'extends' in question_lower:
            # Check if this code extends the target
            if target in metadata.get('extends', []):
                filtered.append(doc)
        elif 'implements' in question_lower:
            # Check if this code implements the target
            if target in metadata.get('implements', []):
                filtered.append(doc)

    return filtered if filtered else documents  # Return all if no matches

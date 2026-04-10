import os
import sys
import logging
from typing import List

# Add the libs/insight_engine folder to path so its internal imports work
ENGINE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "libs/insight_engine"))
if ENGINE_ROOT not in sys.path:
    sys.path.append(ENGINE_ROOT)

from langchain_core.documents import Document
from workflow.builder import build_workflow

logger = logging.getLogger(__name__)

class ObserverRetrieverAdapter:
    """
    Adapter that wraps CodeVectorStore to match the 'retriever' 
    interface expected by the Insight Engine.
    """
    def __init__(self, vector_store, k: int = 3):
        self.vector_store = vector_store
        self.k = k

    def invoke(self, query: str) -> List[Document]:
        """Insight Engine calls .invoke(query)"""
        return self.vector_store.similarity_search(query, k=self.k)

class AdaptiveObserverConsultant:
    """
    Main entry point for using the integrated Insight Engine
    with the Autonomous System Observer.
    """
    def __init__(self, llm, vector_store):
        self.llm = llm
        self.vector_store = vector_store
        
        # Wrap our store in the adapter
        self.retriever = ObserverRetrieverAdapter(vector_store)
        
        # Build the imported workflow
        self.app = build_workflow(self.llm, self.retriever)

    def ask(self, question: str):
        """Invoke the elite-rank reasoning loop"""
        initial_state = {
            "question": question,
            "retry_count": 0,
            "intermediate_steps": []
        }
        
        config = {"configurable": {"thread_id": "observer-consult-v1"}}
        
        # Run the adaptive loop
        final_state = self.app.invoke(initial_state, config=config)
        
        return final_state

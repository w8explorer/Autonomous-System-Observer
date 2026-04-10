"""
System initialization logic
"""

import streamlit as st
from typing import Tuple, Optional, Any

from services import DocumentLoaderService, VectorStoreService, LLMService
from utils import extract_java_metadata
from workflow import build_workflow


@st.cache_resource
def initialize_system(java_project_path: str) -> Tuple[Optional[Any], Optional[Any], int]:
    """
    Initialize the Code RAG system with caching

    Args:
        java_project_path: Path to Java project directory

    Returns:
        Tuple of (compiled_workflow, retriever, file_count)
    """
    # Initialize services
    llm_service = LLMService()
    vectorstore_service = VectorStoreService()
    loader_service = DocumentLoaderService(java_project_path)

    # Load Java files
    with st.spinner("Loading Java code files..."):
        try:
            docs, file_count = loader_service.load_documents()

            if not docs:
                st.warning(f"⚠️ No Java files found in {java_project_path}")
                return None, None, 0

            st.success(f"✅ Loaded {file_count} Java files")

        except Exception as e:
            st.error(f"❌ Error loading files: {e}")
            return None, None, 0

    # Code-aware text splitter
    with st.spinner("Processing code into chunks..."):
        splits = loader_service.split_documents(docs)
        st.success(f"✅ Created {len(splits)} code chunks")

    # Extract relationships and add to metadata (Hybrid Lite!)
    with st.spinner("Extracting code relationships..."):
        for split in splits:
            code_metadata = extract_java_metadata(
                split.page_content,
                split.metadata.get('source', '')
            )
            # Add relationship metadata to each chunk
            split.metadata.update(code_metadata)

        st.success(f"✅ Extracted relationships from {len(splits)} code chunks")

    # Create vector store
    with st.spinner("Building vector store (this may take a moment)..."):
        vectorstore_service.build_vectorstore(splits)
        retriever = vectorstore_service.get_retriever()
        st.success(f"✅ Vector store ready with {len(splits)} embeddings + relationships")

    # Build workflow
    app = build_workflow(llm_service.get_llm(), retriever)

    return app, retriever, file_count

"""
Vector store service for code embeddings
"""

from typing import List
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

from config.settings import OPENAI_EMBEDDING_MODEL, VECTOR_STORE_K


class VectorStoreService:
    """Service for managing vector store and retrieval"""

    def __init__(self):
        self.embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)
        self.vectorstore = None
        self.retriever = None

    def build_vectorstore(self, documents: List[Document]):
        """
        Build FAISS vector store from documents

        Args:
            documents: List of document chunks with metadata
        """
        self.vectorstore = FAISS.from_documents(documents, self.embeddings)
        self.retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": VECTOR_STORE_K}
        )

    def get_retriever(self):
        """Get the retriever instance"""
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call build_vectorstore first.")
        return self.retriever

    def retrieve_documents(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query

        Args:
            query: Search query

        Returns:
            List of relevant documents
        """
        return self.retriever.invoke(query)

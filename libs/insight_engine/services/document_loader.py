"""
Document loading and processing service
"""

from typing import List, Tuple
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from config.settings import CODE_SEPARATORS, CHUNK_SIZE, CHUNK_OVERLAP


class DocumentLoaderService:
    """Service for loading and processing code documents"""

    def __init__(self, java_project_path: str):
        self.java_project_path = java_project_path
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=CODE_SEPARATORS
        )

    def load_documents(self) -> Tuple[List[Document], int]:
        """
        Load Java files from the project directory

        Returns:
            Tuple of (documents list, file count)
        """
        loader = DirectoryLoader(
            self.java_project_path,
            glob="**/*.java",
            loader_cls=TextLoader,
            show_progress=False,
            use_multithreading=True
        )
        docs = loader.load()
        return docs, len(docs)

    def split_documents(self, docs: List[Document]) -> List[Document]:
        """
        Split documents into chunks using code-aware splitting

        Args:
            docs: List of documents to split

        Returns:
            List of document chunks
        """
        return self.text_splitter.split_documents(docs)

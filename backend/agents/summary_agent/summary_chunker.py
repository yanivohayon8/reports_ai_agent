from pathlib import Path
from typing import List
import sys
import os
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Add the project root to Python path for direct execution
project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.core.pdf_reader import read_pdf
from backend.indexer.indexer import FAISSIndexer


class SummaryChunker:
    """
    Splits PDF documents into chunks using RecursiveCharacterTextSplitter,
    and audits the results into FAISSIndexer.
    """

    def __init__(self, faiss_indexer: FAISSIndexer, text_splitter: RecursiveCharacterTextSplitter):
        self.faiss_indexer = faiss_indexer
        self.text_splitter = text_splitter

    def chunk(self, pdf_path: Path) -> List[Document]:
        chunks_doc_processed = self._chunk_text(pdf_path)
        self._audit_indexer(pdf_path, chunks_doc_processed)
        return chunks_doc_processed

    def _chunk_text(self, pdf_path: Path) -> List[Document]:
        # Read PDF as continuous text
        text = read_pdf(pdf_path, format="text", split_by_page=False)

        # Split into chunks
        text_chunks = self.text_splitter.split_text(text)

        # Wrap chunks as Documents with metadata
        documents = [
            Document(
                page_content=chunk,
                metadata={
                    "source": str(pdf_path),
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                },
            )
            for i, chunk in enumerate(text_chunks)
        ]

        return documents

    def _audit_indexer(self, pdf_path: Path, chunks_doc_processed: List[Document]):
        self.faiss_indexer.add_documents(chunks_doc_processed)
        self.faiss_indexer.audit_processed_pdf(pdf_path)
        self.faiss_indexer.audit_splitter(self.text_splitter)

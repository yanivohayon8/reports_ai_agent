from indexer.indexer import FAISSIndexer
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class SummaryChunker:
    def __init__(self, faiss_indexer:FAISSIndexer, text_splitter:RecursiveCharacterTextSplitter):
        self.faiss_indexer = faiss_indexer
        self.text_splitter = text_splitter # You should use larger chunk size for summary

    def chunk(self, pdf_path: Path):
        chunks_doc_processed = self._chunk_text(pdf_path)

        self._audit_indexer(pdf_path,chunks_doc_processed)
        
    def _chunk_text(self, pdf_path: Path)->list[Document]:
        pass

    def _audit_indexer(self, pdf_path: Path,chunks_doc_processed:list[Document]):
        self.faiss_indexer.add_documents(chunks_doc_processed)
        self.faiss_indexer.audit_processed_pdf(pdf_path)
        self.faiss_indexer.audit_splitter(self.text_splitter)
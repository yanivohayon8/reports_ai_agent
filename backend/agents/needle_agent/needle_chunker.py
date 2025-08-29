import sys
import os
from indexer.indexer import FAISSIndexer
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from backend.core.pdf_reader import read_pdf

class NeedleChunker:
    def __init__(self, faiss_indexer:FAISSIndexer,text_splitter:RecursiveCharacterTextSplitter):
        self.faiss_indexer = faiss_indexer
        self.text_splitter = text_splitter # You should use smaller chunk size for needle

    def chunk(self, pdf_path: Path):
        pages = read_pdf(pdf_path,format="documents")
        chunks_within_pages = self._chunk_within_each_page(pages)

        chunks_doc_processed = []

        for chunk in chunks_within_pages:
            # TODO: remove the table of contents from the chunk?
            metadata = self._process_metadata(chunk)
            chunks_doc_processed.append(Document(chunk.page_content,metadata=metadata))
        
        self.faiss_indexer.add_documents(chunks_doc_processed)
        self._audit_indexer(pdf_path)

    def _chunk_within_each_page(self, pages:list[Document])->list[Document]:
        return self.text_splitter.split_documents(pages)
    
    def _process_metadata(self,chunk:Document)->dict:
        metadata = {}
        metadata["page_number"] = chunk.metadata.get("page")

        # TODO: if necessary - add more metadata here

        return metadata

    def _audit_indexer(self, pdf_path: Path):
        self.faiss_indexer.audit_processed_pdf(pdf_path)
        self.faiss_indexer.audit_splitter(self.text_splitter)

    def save(self,directory_path:Path):
        self.faiss_indexer.save(directory_path)

    
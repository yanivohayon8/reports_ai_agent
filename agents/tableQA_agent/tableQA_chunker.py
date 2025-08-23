from indexer.indexer import FAISSIndexer
from pathlib import Path
from langchain_core.documents import Document
from core.pdf_reader import extract_tables
import pandas as pd
from langchain_core.language_models import BaseChatModel
from agents.tableQA_agent.tableQA_prompts import table_summary_template
from enum import Enum

class TableChunkType:
    DESCRIPTION = "description"
    RAW = "raw"
    

class TableQAChunker:
    def __init__(self, faiss_indexer:FAISSIndexer, llm: BaseChatModel):
        self.faiss_indexer = faiss_indexer
        self.llm = llm

    def chunk(self, pdf_path: Path):
        tables = extract_tables(pdf_path)
        chunks = []

        for table_index,table in enumerate(tables):
            markdown_table = table.to_markdown(index=False)

            common_metadata = {}
            common_metadata["table_index"] = table_index
            common_metadata["source"] = pdf_path

            raw_chunk_metadata = common_metadata.copy()
            raw_chunk_metadata["table_chunk_type"] = TableChunkType.RAW
            raw_chunk = Document(markdown_table,metadata=raw_chunk_metadata)
            chunks.append(raw_chunk)

            description = self._get_table_description(markdown_table)
            description_chunk_metadata = common_metadata.copy()
            description_chunk_metadata["table_chunk_type"] = TableChunkType.DESCRIPTION
            description_chunk = Document(description,metadata=description_chunk_metadata)
            chunks.append(description_chunk)

        self.faiss_indexer.add_documents(chunks)

        self._audit_indexer(pdf_path)

    def _get_table_description(self,markdown_table:str)->str:
        # TODO: Implement a way to get the table description
        prompt = table_summary_template.invoke({"markdown_table":markdown_table})
        response = self.llm.invoke(prompt)

        return response.content
        
    def _audit_indexer(self, pdf_path: Path):
        self.faiss_indexer.audit_processed_pdf(pdf_path)


    def save(self,directory_path:Path):
        self.faiss_indexer.save(directory_path)

    
    def get_used_input(self)->dict:
        faiss_indexer_input = self.faiss_indexer.get_used_input()

        return {
            "faiss_indexer_input": faiss_indexer_input,
            "llm": self.llm._identifying_params
        }
import pytest
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.agents.summary_agent.summary_chunker import SummaryChunker
from backend.indexer.indexer import FAISSIndexer


def test_chunk_text_from_pdf(tmp_path):
    # create a dummy PDF as a text file (in practice we use the pdf_reader from the project)
    pdf_file = tmp_path / "sample.txt"
    pdf_file.write_text("This is a test document.\nIt has multiple lines.\nAnd should be chunked.", encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(chunk_size=30, chunk_overlap=5)
    faiss_indexer = FAISSIndexer.from_small_embedding(directory_path=str(tmp_path))

    chunker = SummaryChunker(faiss_indexer=faiss_indexer, text_splitter=splitter)

    chunks = chunker._chunk_text(pdf_file)

    assert isinstance(chunks, list)
    assert all(hasattr(doc, "page_content") for doc in chunks)
    assert len(chunks) > 0


def test_chunk_and_audit(tmp_path):
    pdf_file = tmp_path / "sample.txt"
    pdf_file.write_text("Another test document with more lines.\nLine 2.\nLine 3.", encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(chunk_size=25, chunk_overlap=5)
    faiss_indexer = FAISSIndexer.from_small_embedding(directory_path=str(tmp_path))

    chunker = SummaryChunker(faiss_indexer=faiss_indexer, text_splitter=splitter)

    chunks = chunker.chunk(pdf_file)

    # verify that the chunks are saved in the index
    used_input = faiss_indexer.get_used_input()
    assert "processed_pdfs" in used_input
    assert str(pdf_file) in used_input.get("processed_pdfs", [])

import os
import shutil
from pathlib import Path
from langchain_core.documents import Document

import sys

sys.path.append(str(Path(__file__).parent.parent))

from core.indexer import FAISSIndexer,TextChunker

def test_faiss_indexer():
    indexer = FAISSIndexer.from_small_embedding()

    documents = [
        Document(page_content="Hello, world!", metadata={"source": "test1"}),
        Document(page_content="This is a test document.", metadata={"source": "test2"}),
    ]

    indexer.add_documents(documents)
    directory_path = os.path.join("tests","data","temp_faiss_index")
    indexer.save(directory_path)

    # delete the index directory and all its contents
    shutil.rmtree(directory_path)


def test_text_chunker():
    text_chunker = TextChunker.from_pdf(Path("tests/data/report.pdf"),Path("tests","data","temp_faiss_index"))
    chunks = text_chunker._chunk_text()
    assert len(chunks) > 0
    
    print(chunks[0])


def test_text_chunker_chunk():
    pdf_path = Path("tests/data/report.pdf")
    faiss_indexer_directory = Path("tests","data","temp_faiss_index")
    text_chunker = TextChunker.from_pdf(pdf_path,faiss_indexer_directory)

    text_chunker.chunk(pdf_path)
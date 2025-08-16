import os
import shutil
from pathlib import Path
from langchain_core.documents import Document

import sys

sys.path.append(str(Path(__file__).parent.parent))

from indexer.indexer import FAISSIndexer,TextChunker
from core.text_splitter import get_text_splitter

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


def test_text_chunker_chunk():
    faiss_indexer_directory = Path("tests","data","temp_faiss_index")
    faiss_indexer = FAISSIndexer.from_small_embedding(directory_path=faiss_indexer_directory)
    text_splitter = get_text_splitter()
    text_chunker = TextChunker(faiss_indexer,text_splitter)

    pdf_path = Path("tests/data/report.pdf")
    text_chunker.chunk(pdf_path)
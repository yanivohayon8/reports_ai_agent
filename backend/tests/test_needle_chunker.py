from agents.needle_agent.needle_chunker import NeedleChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from indexer.indexer import FAISSIndexer
from pathlib import Path
from core.api_utils import get_openai_embeddings
import os
import shutil

def test_needle_chunker_compiles():
    faiss_indexer = FAISSIndexer(get_openai_embeddings(model="text-embedding-3-small"))
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=10)

    chunker = NeedleChunker(faiss_indexer,text_splitter)
    chunker.chunk(Path("tests/data/report.pdf"))

    temp_directory_path = os.path.join("tests","data","temp_faiss_index")
    
    faiss_indexer.save(temp_directory_path)

    # load and test the index
    faiss_indexer_loaded = FAISSIndexer(get_openai_embeddings(model="text-embedding-3-small"),
                                        directory_path=temp_directory_path)
    assert faiss_indexer_loaded is not None

    # test a chunk and its metadata
    chunk = faiss_indexer_loaded.retrieve(query="What is the main finding of the report?",
                                          num_documents=1)
    chunk = chunk[0]
    assert chunk is not None
    assert chunk.metadata is not None
    assert chunk.metadata["page_number"] is not None
    
    shutil.rmtree(temp_directory_path)




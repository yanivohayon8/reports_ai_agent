from agents.tableQA_agent.tableQA_chunker import TableQAChunker
from core.api_utils import get_openai_embeddings,get_llm_langchain_openai
from indexer.indexer import FAISSIndexer
from pathlib import Path
import shutil

def test_tableQA_chunker():
    embeddings = get_openai_embeddings(model="text-embedding-3-small")
    faiss_indexer = FAISSIndexer(embeddings)
    llm = get_llm_langchain_openai(model="gpt-4o-mini")
    tableQA_chunker = TableQAChunker(faiss_indexer,llm)

    tableQA_chunker.chunk(Path("tests/data/client2_report2_tourAndCarePolicy.pdf"))
    
    temp_dir = Path("tests/data/temp_tableQA_chunker")
    tableQA_chunker.save(temp_dir)

    # load the indexer
    tableQA_indexer = FAISSIndexer(embeddings, directory_path=temp_dir)
    
    documents = tableQA_indexer.retrieve("What is the total expenses?",num_documents=2)
    
    assert len(documents) == 2
    print(documents[0].metadata["table_chunk_type"])
    print(documents[1].metadata["table_chunk_type"])

    # delete the temp directory
    shutil.rmtree(temp_dir)



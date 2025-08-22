from agents.needle_agent.needle_agent import NeedleAgent
from indexer.indexer import FAISSIndexer
from core.api_utils import get_llm_langchain_openai


def test_agent_compiles():
    faiss_indexer = FAISSIndexer.from_small_embedding(directory_path="vectordb_indexes/faiss_indexer_insurance")
    llm = get_llm_langchain_openai(model="gpt-4o-mini")
    needle_agent = NeedleAgent(faiss_indexer,llm)

    query = "What happened to Alex from Canada?"
    answer = needle_agent.answer(query)

    print("-"*50 + "answer" + "-"*50)
    print("\t" + answer["answer"])

    print("\t" + "-"*50 + "chunks" + "-"*50)
    
    for chunk_content,chunk_metadata in zip(answer["chunks_content"],answer["chunks_metadata"]):
        print("\t" + "-"*50 + "page_content" + "-"*50)
        print("\t" + chunk_content)
        print("\t" + "-"*50 + "metadata" + "-"*50)
        print("\t" + str(chunk_metadata))

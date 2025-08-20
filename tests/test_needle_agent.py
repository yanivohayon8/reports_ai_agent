from needle_agent.needle import NeedleAgent
from indexer.indexer import FAISSIndexer
from core.api_utils import get_llm_langchain_openai


def test_needle_agent():
    faiss_indexer = FAISSIndexer.from_small_embedding(directory_path="vectordb_indexes/faiss_indexer_insurance")
    llm = get_llm_langchain_openai(model="gpt-4o-mini")
    needle_agent = NeedleAgent(faiss_indexer,llm)

    query = "Who is the main person in the report?"
    answer = needle_agent.answer(query)

    print("-"*50 + "answer" + "-"*50)
    print(answer["answer"])

    print("-"*50 + "chunks" + "-"*50)
    print(answer["chunks"])
    
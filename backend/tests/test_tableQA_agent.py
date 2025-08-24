from agents.tableQA_agent.tableQA_agent import TableQAgent
from indexer.indexer import FAISSIndexer
from core.api_utils import get_llm_langchain_openai,get_openai_embeddings


def test_retrieve_context():
    # TODO: create a faiss indexer with a test pdf
    faiss_indexer = FAISSIndexer(get_openai_embeddings(model="text-embedding-3-small"),directory_path="vectordb_indexes/tableQA")
    llm = get_llm_langchain_openai(model="gpt-4o-mini")
    agent = TableQAgent(faiss_indexer,llm)

    query = "What is the total amount of the policy?"
    context,chunks = agent._retrieve_context(query)

    assert context is not None
    assert chunks is not None

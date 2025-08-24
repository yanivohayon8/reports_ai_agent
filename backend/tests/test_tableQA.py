import pytest
from langchain_core.documents import Document

from agents.tableQA_agent.tableQA import TableQAgent


class MockRetriever:
    """A fake retriever that returns predefined documents."""
    def __init__(self, docs):
        self.docs = docs

    def retrieve(self, query: str, k_dense: int = 3, k_sparse: int = 3):
        return self.docs


@pytest.mark.asyncio
async def test_tableqa_with_table(monkeypatch):
    # Fake docs containing a table
    docs = [
        Document(page_content="| Name | Salary |\n| Alice | 5000 |\n| Bob | 7000 |"),
    ]

    retriever = MockRetriever(docs)

    # Patch LLMChain.invoke to avoid real API calls
    async def fake_invoke(self, inputs):
        return {"text": f"FAKE_ANSWER: {inputs['query']} → using {len(inputs['table'])} chars of table"}

    monkeypatch.setattr("agents.tableQA_agent.tableQA.LLMChain.invoke", fake_invoke)

    agent = TableQAgent(retriever=retriever)
    result = await agent.handle("Who has the highest salary?")
    assert "FAKE_ANSWER" in result
    assert "Who has the highest salary?" in result


@pytest.mark.asyncio
async def test_tableqa_without_table(monkeypatch):
    # Docs without any table-like structure
    docs = [Document(page_content="This is plain text without table.")]

    retriever = MockRetriever(docs)

    agent = TableQAgent(retriever=retriever)
    result = await agent.handle("Which row has the max value?")
    assert result == "No relevant table found in the documents."

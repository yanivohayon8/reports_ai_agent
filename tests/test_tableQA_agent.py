import pytest
from types import SimpleNamespace

from backend.agents.tableQA_agent.tableQA_agent import TableQAgent


class MockLLM:
    def __init__(self):
        self._identifying_params = {"model": "mock-llm"}

    def invoke(self, inputs):
        # Simulate a minimal LangChain ChatModel response object
        return SimpleNamespace(content=f"Answer based on table: {inputs.get('table','')[:20]} ... for query: {inputs.get('query','')}")

    def __or__(self, other):
        # Support TABLE_QA_PROMPT | llm chaining by returning self
        return self


class MockDocument:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata


class MockRetriever:
    def __init__(self, documents):
        self._documents = documents

    def retrieve(self, query, k_dense=5, k_sparse=5):
        return list(self._documents)


@pytest.fixture
def sample_docs():
    # Description and RAW for table 0
    d0 = MockDocument(
        page_content="Coverage for medical expenses",
        metadata={"table_chunk_type": "description", "table_index": 0, "source": "policy.pdf"},
    )
    r0a = MockDocument(
        page_content="| Item | Limit |\n| A | 100 |",
        metadata={"table_chunk_type": "raw", "table_index": 0, "source": "policy.pdf"},
    )
    r0b = MockDocument(
        page_content="| Item | Deductible |\n| A | 10 |",
        metadata={"table_chunk_type": "raw", "table_index": 0, "source": "policy.pdf"},
    )

    # Description and RAW for table 1
    d1 = MockDocument(
        page_content="Premium table for customers",
        metadata={"table_chunk_type": "description", "table_index": 1, "source": "policy.pdf"},
    )
    r1 = MockDocument(
        page_content="| Customer | Premium |\n| Bob | 30 |",
        metadata={"table_chunk_type": "raw", "table_index": 1, "source": "policy.pdf"},
    )

    return [d0, r0a, r0b, d1, r1]


def test_answer_selects_related_raw_chunks(sample_docs):
    agent = TableQAgent(retriever=MockRetriever(sample_docs), llm=MockLLM())
    result = agent.answer("What is the deductible?")

    assert result["answer"]
    assert len(result["chunks_content"]) >= 3  # description + raw rows for table 0
    assert any("Deductible" in c or "deductible" in c for c in result["chunks_content"])

    for md in result["chunks_metadata"]:
        assert md["source"] == "policy.pdf"
        assert md["table_index"] == 0


def test_no_description_returns_no_table():
    docs = [
        MockDocument(
            page_content="| X | Y |\n| 1 | 2 |",
            metadata={"table_chunk_type": "raw", "table_index": 2, "source": "foo.pdf"},
        )
    ]
    agent = TableQAgent(retriever=MockRetriever(docs), llm=MockLLM())
    result = agent.answer("anything")

    assert result["answer"] == "No relevant table found."
    assert result["chunks_content"] == []
    assert result["chunks_metadata"] == []


def test_handle_wraps_answer_and_includes_agent(sample_docs):
    agent = TableQAgent(retriever=MockRetriever(sample_docs), llm=MockLLM())

    import asyncio
    out = asyncio.get_event_loop().run_until_complete(agent.handle("Show premiums"))

    assert out["agent"] == "TableQAgent"
    assert "chunks_content" in out and "chunks_metadata" in out
    assert isinstance(out["answer"], str) and len(out["answer"]) > 0


def test_get_used_input_exposes_identifying_params(sample_docs):
    agent = TableQAgent(retriever=MockRetriever(sample_docs), llm=MockLLM())
    used = agent.get_used_input()

    assert used["retriever"] == "HybridRetriever"
    assert used["llm"]["model"] == "mock-llm"

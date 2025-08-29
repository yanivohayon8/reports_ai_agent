import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest

# Ensure backend package is importable
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from agents.needle_agent.needle_agent import NeedleAgent
from langchain_core.documents import Document


class MockFAISSIndexer:
    def __init__(self, retrieved_docs):
        self._retrieved_docs = retrieved_docs

    def retrieve(self, query: str):
        return list(self._retrieved_docs)

    def get_used_input(self):
        return {"index": "faiss_indexer_insurance", "dim": 1536}


class MockLLM:
    def __init__(self, reply_text: str = "Mock answer"):
        self.reply_text = reply_text
        self._identifying_params = {"model": "gpt-4o-mini"}

    def invoke(self, prompt):
        return SimpleNamespace(content=self.reply_text)


def make_docs(n=2):
    return [
        Document(page_content=f"content-{i}", metadata={"source": f"s{i}", "page": i + 1})
        for i in range(n)
    ]


def test_answer_returns_expected_fields():
    docs = make_docs(2)
    indexer = MockFAISSIndexer(docs)
    llm = MockLLM("final-answer")

    agent = NeedleAgent(indexer, llm)
    result = agent.answer("where is the fact?")

    assert result["answer"] == "final-answer"
    assert result["chunks_content"] == [d.page_content for d in docs]
    assert result["chunks_metadata"] == [d.metadata for d in docs]


@pytest.mark.asyncio
async def test_handle_wraps_answer_with_metadata():
    docs = make_docs(1)
    indexer = MockFAISSIndexer(docs)
    llm = MockLLM("ok")
    agent = NeedleAgent(indexer, llm)

    out = await agent.handle("q")
    assert out["answer"] == "ok"
    assert out["agent"] == "Needle Agent"
    assert "reasoning" in out


def test_concat_chunks_concatenates_with_separators():
    docs = make_docs(3)
    indexer = MockFAISSIndexer(docs)
    llm = MockLLM()
    agent = NeedleAgent(indexer, llm)

    context = agent._concat_chunks(docs)
    assert context.count("\n\n") == 2
    assert all(d.page_content in context for d in docs)


def test_retrieve_context_returns_context_and_chunks():
    docs = make_docs(2)
    indexer = MockFAISSIndexer(docs)
    llm = MockLLM()
    agent = NeedleAgent(indexer, llm)

    context, chunks = agent._retrieve_context("q")
    assert isinstance(context, str)
    assert chunks == docs


def test_generate_returns_llm_content():
    docs = make_docs(1)
    indexer = MockFAISSIndexer(docs)
    llm = MockLLM("generated")
    agent = NeedleAgent(indexer, llm)

    out = agent._generate("ctx", "q")
    assert out == "generated"


def test_get_used_input_contains_llm_and_indexer_details():
    indexer = MockFAISSIndexer(make_docs(1))
    llm = MockLLM()
    agent = NeedleAgent(indexer, llm)

    used = agent.get_used_input()
    assert "faiss_indexer_input" in used
    assert "llm_model" in used
    assert used["llm_model"]["model"] == "gpt-4o-mini"

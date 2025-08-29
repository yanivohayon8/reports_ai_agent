import pytest
from llama_index.core import Document
from backend.indexer.graph_indexer import GraphIndexer


def test_graph_indexer_add_and_build():
    graph_indexer = GraphIndexer()

    # add dummy documents (using keyword arg "text")
    docs_policy = [Document(text="Policy: Covers medical expenses.")]
    docs_reports = [Document(text="Medical report: Patient admitted to hospital.")]

    graph_indexer.add_documents(docs_policy, index_name="policies")
    graph_indexer.add_documents(docs_reports, index_name="reports")

    # build graph
    graph_indexer.build_graph()

    metadata = graph_indexer.get_used_input()
    assert metadata["nodes"] == 2
    assert "policies" in metadata["indices"]
    assert "reports" in metadata["indices"]

    result = graph_indexer.retrieve("What does the policy cover?")
    assert result is not None
    assert "Policy" in str(result) or "medical" in str(result)


def test_graph_indexer_fails_without_build():
    graph_indexer = GraphIndexer()
    with pytest.raises(RuntimeError):
        graph_indexer.retrieve("dummy question")

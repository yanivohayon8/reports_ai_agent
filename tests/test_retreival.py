from langchain_core.documents import Document
from backend.indexer.indexer import FAISSIndexer
from backend.retrieval.dense_retriever import DenseRetriever
from backend.retrieval.sparse_retriever import SparseRetriever
from backend.retrieval.hybrid_retriever import HybridRetriever


def test_dense_retriever():
    docs = [
        Document(page_content="Insurance policy 123"),
        Document(page_content="Claim filed March 2025"),
    ]
    indexer = FAISSIndexer.from_small_embedding()
    indexer.add_documents(docs)

    retriever = DenseRetriever(indexer.vector_store)
    results = retriever.retrieve("policy", k=1)
    assert len(results) > 0
    assert "policy" in results[0].page_content


def test_sparse_retriever():
    docs = [
        Document(page_content="Insurance policy 123"),
        Document(page_content="Claim filed March 2025"),
    ]
    retriever = SparseRetriever(docs)
    results = retriever.retrieve("Claim", k=1)
    assert len(results) > 0
    assert "Claim" in results[0].page_content


def test_hybrid_retriever():
    docs = [
        Document(page_content="Insurance policy 123"),
        Document(page_content="Claim filed March 2025"),
    ]
    indexer = FAISSIndexer.from_small_embedding()
    indexer.add_documents(docs)

    dense = DenseRetriever(indexer.vector_store)
    sparse = SparseRetriever(docs)
    hybrid = HybridRetriever(dense, sparse)

    results = hybrid.retrieve("policy", k_dense=1, k_sparse=1)
    assert len(results) > 0
    assert any("policy" in r.page_content for r in results)

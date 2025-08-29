import sys
import os

# Add the backend directory to Python path
backend_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, backend_path)

from typing import List
from langchain_core.documents import Document
from retrieval.dense_retriever import DenseRetriever
from retrieval.sparse_retriever import SparseRetriever

class HybridRetriever:
    def __init__(self, dense: DenseRetriever, sparse: SparseRetriever):
        self.dense = dense
        self.sparse = sparse
    def retrieve(self, query: str, k_dense: int = 5, k_sparse: int = 5) -> List[Document]:
        dense_results = self.dense.retrieve(query, k=k_dense)
        sparse_results = self.sparse.retrieve(query, k=k_sparse)
        results = {id(doc): doc for doc in dense_results + sparse_results}
        return list(results.values())

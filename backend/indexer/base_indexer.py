from abc import ABC, abstractmethod
from typing import List, Any

class BaseIndexer(ABC):
    @abstractmethod
    def add_documents(self, documents: List[Any]):
        """Add new documents to the index"""
        pass

    @abstractmethod
    def retrieve(self, query: str, **kwargs):
        """Retrieve relevant documents given a query"""
        pass

    @abstractmethod
    def save(self, directory_path: str):
        """Persist the index to disk"""
        pass

    @abstractmethod
    def get_used_input(self) -> dict:
        """Return metadata or config used by this indexer"""
        pass

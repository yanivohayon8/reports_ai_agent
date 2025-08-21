from typing import List, Dict, Any
from llama_index.core import Document

class HybridRetriever:
    """Simple retriever that returns documents based on basic text matching."""

    def __init__(self, documents: List[Document]):
        self.documents = documents

    def retrieve(self, query: str, filters: Dict[str, Any] = None, top_k: int = 8):
        """Simple retrieval based on document text content."""
        # For now, just return documents that contain query terms
        query_terms = query.lower().split()
        scored_docs = []
        
        for doc in self.documents:
            doc_text = doc.text.lower()
            score = sum(1 for term in query_terms if term in doc_text)
            if score > 0:
                scored_docs.append((score, doc))
        
        # Sort by score and return top_k
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        results = [doc for score, doc in scored_docs[:top_k]]
        
        # Apply filters if provided
        if filters:
            def matches_filters(doc):
                return all(doc.metadata.get(k) == v for k, v in filters.items())
            results = [doc for doc in results if matches_filters(doc)]
        
        return results

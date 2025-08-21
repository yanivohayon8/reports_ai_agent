from langchain_community.vectorstores import FAISS


class DenseRetriever:
    def __init__(self, faiss_index: FAISS):
        self.faiss_index = faiss_index

    def retrieve(self, query: str, k: int = 5):
        try:
            index = getattr(self.faiss_index, "index", None)
            if index is not None and getattr(index, "ntotal", 0) <= 0:
                return []
        except Exception:
            pass
        return self.faiss_index.similarity_search(query, k=k)

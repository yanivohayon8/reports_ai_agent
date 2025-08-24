from rank_bm25 import BM25Okapi
from langchain_core.documents import Document
import re

class SparseRetriever:
    def __init__(self, corpus: list[Document]):
        self.corpus = corpus
        self.tokenized = [self._tokenize(doc.page_content) for doc in corpus]
        self.bm25 = BM25Okapi(self.tokenized)

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def retrieve(self, query: str, k: int = 5):
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        ranked = list(zip(self.corpus, self.tokenized, scores))
        # If BM25 yields ties everywhere, break ties by overlap count
        if ranked and max(s for _, _, s in ranked) == min(s for _, _, s in ranked):
            ranked.sort(
                key=lambda x: len(set(query_tokens) & set(x[1])),
                reverse=True,
            )
        else:
            ranked.sort(key=lambda x: x[2], reverse=True)
        return [doc for doc, _, _ in ranked[:k]]

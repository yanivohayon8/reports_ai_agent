from langchain_core.language_models.chat_models import BaseChatModel
from retrieval.hybrid_retriever import HybridRetriever


class TableQAgent:
    def __init__(self, retriever: HybridRetriever, llm: BaseChatModel):
        self.retriever = retriever
        self.llm = llm

    def answer(self, query: str) -> str:
        pass

    def _retrieve_context(self, query: str) -> str:
        pass
    
    def _generate(self, context: str, query: str) -> str:
        pass

    def get_used_input(self) -> dict:
        pass


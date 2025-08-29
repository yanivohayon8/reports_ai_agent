import sys
import os
from typing import Dict

# Add the backend directory to Python path
backend_path = os.path.join(os.path.dirname(__file__), '..', '..')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from langchain_core.language_models.chat_models import BaseChatModel
from backend.indexer.graph_indexer import GraphIndexer
from backend.agents.summary_agent.summary_prompts import QUERY_BASED_SUMMARY_PROMPT_CHAT_TEMPLATE


class SummaryAgentGraph:
    """
    SummaryAgent variant that uses GraphIndexer instead of FAISS + Chunker.
    Provides summaries or contextual overviews directly from the Graph RAG.
    """

    def __init__(self, graph_indexer: GraphIndexer, llm: BaseChatModel = None) -> None:
        """
        :param graph_indexer: GraphIndexer object (already built/loaded)
        :param llm: optional LLM (kept for compatibility)
        """
        self.graph_indexer = graph_indexer
        self.llm = llm

    async def handle(self, query: str) -> Dict:
        """
        Async entry point for RouterAgent.
        Executes a query and returns structured response.
        """
        result = self.answer(query)
        return {
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "agent": "Summary Agent (Graph RAG)",
            "reasoning": "Summary generated from Graph RAG retrieval"
        }

    def answer(self, query: str) -> Dict:
        """
        Run a query on GraphIndexer and return both the answer and sources.
        """
        try:
            result = self.graph_indexer.retrieve(query)
        except Exception as e:
            return {
                "answer": f"Graph retrieval failed: {e}",
                "sources": []
            }

        # If we have an external LLM and sources, compose a focused summary for better quality
        try:
            sources = result.get("sources", []) or []
            if self.llm and sources:
                context_text = "\n\n".join([s.get("snippet", "") for s in sources if s.get("snippet")])
                if context_text.strip():
                    chain = QUERY_BASED_SUMMARY_PROMPT_CHAT_TEMPLATE | self.llm
                    resp = chain.invoke({"query": query, "document": context_text})
                    composed_answer = getattr(resp, "content", str(resp))
                    return {"answer": composed_answer, "sources": sources}
        except Exception:
            # Fall back to graph's direct answer on any error
            pass

        return {"answer": result.get("answer", "No answer found."), "sources": result.get("sources", [])}

    def get_used_input(self) -> dict:
        """
        Return metadata about the resources used by this agent.
        """
        return {
            "graph_indexer_input": self.graph_indexer.get_used_input(),
            "llm_model": getattr(self.llm, "_identifying_params", "unknown") if self.llm else None
        }

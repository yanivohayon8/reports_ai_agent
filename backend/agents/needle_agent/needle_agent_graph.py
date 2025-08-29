import sys
import os
from typing import Dict

# Add the backend directory to Python path
backend_path = os.path.join(os.path.dirname(__file__), '..', '..')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from langchain_core.language_models.chat_models import BaseChatModel
from backend.indexer.graph_indexer import GraphIndexer


class NeedleAgentGraph:
    """
    NeedleAgent variant that uses GraphIndexer instead of FAISS.
    Focused on precise fact-level retrieval with sources.
    """

    def __init__(self, graph_indexer: GraphIndexer, llm: BaseChatModel) -> None:
        """
        Initialize NeedleAgent with GraphIndexer and LLM.
        :param graph_indexer: GraphIndexer object (already built/loaded)
        :param llm: Language model (kept for compatibility, can be used for reasoning if needed)
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
            "chunks_content": result.get("chunks_content", []),
            "chunks_metadata": result.get("chunks_metadata", []),
            "agent": "Needle Agent (Graph RAG)",
            "reasoning": "Direct fact retrieval using Graph RAG"
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
                "sources": [],
                "chunks_content": [],
                "chunks_metadata": []
            }

        # Extract chunks_content from sources for evaluator compatibility
        sources = result.get("sources", [])
        chunks_content = []
        chunks_metadata = []
        
        if sources:
            # Convert sources to chunks_content format expected by evaluator
            for source in sources:
                if isinstance(source, dict):
                    chunks_content.append(source.get("content", str(source)))
                    chunks_metadata.append(source.get("metadata", {}))
                else:
                    chunks_content.append(str(source))
                    chunks_metadata.append({})

        return {
            "answer": result.get("answer", "No answer found."),
            "sources": sources,
            "chunks_content": chunks_content,
            "chunks_metadata": chunks_metadata
        }

    def get_used_input(self) -> dict:
        """
        Return metadata about the resources used by this agent.
        """
        return {
            "graph_indexer_input": self.graph_indexer.get_used_input(),
            "llm_model": getattr(self.llm, "_identifying_params", "unknown")
        }

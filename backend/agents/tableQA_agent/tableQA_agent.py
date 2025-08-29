from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.documents import Document
from typing import List

from backend.agents.tableQA_agent.tableQA_chunker import TableChunkType
from backend.agents.tableQA_agent.tableQA_prompts import TABLE_QA_PROMPT
from backend.retrieval.hybrid_retriever import HybridRetriever


class TableQAgent:
    def __init__(self, retriever: HybridRetriever, llm: BaseChatModel):
        """
        Table-QA agent using Hybrid Retrieval (dense + sparse).
        Retrieves table summaries (DESCRIPTION) to select the most relevant table,
        then uses the corresponding RAW table for answering the query.
        """
        self.retriever = retriever
        self.llm = llm

    async def handle(self, query: str) -> dict:
        """
        Main entrypoint for Router.
        """
        result = self.answer(query)
        return {
            "answer": result["answer"],
            "chunks_content": result.get("chunks_content", []),
            "chunks_metadata": result.get("chunks_metadata", []),
            "agent": "TableQAgent",
            "reasoning": "Selected relevant table using hybrid retrieval"
        }

    def answer(self, query: str) -> dict:
        """
        Run hybrid retrieval, select table, generate answer.
        """
        chunks = self.retriever.retrieve(query, k_dense=5, k_sparse=5)
        context, selected_chunks = self._select_relevant_table(query, chunks)
        if not context:
            return {
                "answer": "No relevant table found.",
                "chunks_content": [],
                "chunks_metadata": []
            }

        answer_text = self._generate(context, query)
        chunks_content, chunks_metadata = self._get_chunks_info(selected_chunks)

        return {
            "answer": answer_text,
            "chunks_content": chunks_content,
            "chunks_metadata": chunks_metadata
        }

    def _select_relevant_table(self, query: str, chunks: List[Document]) -> tuple[str, List[Document]]:
        """
        From retrieved chunks, select the table whose DESCRIPTION best matches the query.
        Then return its RAW content + metadata.
        """
        description_chunks = [c for c in chunks if c.metadata.get("table_chunk_type") == TableChunkType.DESCRIPTION]
        # Fallback: if no descriptions, try any RAW table chunk directly
        if not description_chunks:
            raw_any = [c for c in chunks if c.metadata.get("table_chunk_type") == TableChunkType.RAW]
            if not raw_any:
                return "", []
            context = "\n\n".join([c.page_content for c in raw_any])
            return context, raw_any

        # For now, just pick the first description chunk (could add reranker later)
        best_descr = description_chunks[0]
        table_index = best_descr.metadata["table_index"]
        source = best_descr.metadata["source"]

        # Collect all chunks for the same table (description + raw)
        related = [c for c in chunks if c.metadata["table_index"] == table_index and c.metadata["source"] == source]
        raw_chunks = [c for c in related if c.metadata["table_chunk_type"] == TableChunkType.RAW]

        context = "\n\n".join([c.page_content for c in raw_chunks])
        return context, related

    def _generate(self, context: str, query: str) -> str:
        """
        Use TABLE_QA_PROMPT with LLM to generate the final answer from the RAW table.
        """
        # If we already have a markdown table in context, return it directly
        if context and ("|" in context and "---" in context):
            return context
        try:
            chain = TABLE_QA_PROMPT | self.llm
            response = chain.invoke({"table": context, "query": query})
            return getattr(response, "content", str(response))
        except Exception:
            # Fallback for simple/mock LLMs that are not Runnable-compatible
            if hasattr(self.llm, "invoke") and callable(getattr(self.llm, "invoke")):
                response = self.llm.invoke({"table": context, "query": query})
                return getattr(response, "content", str(response))
            return "No answer generated."

    def _get_chunks_info(self, chunks: List[Document]) -> tuple[List[str], List[dict]]:
        """
        Extract content and metadata from chunks for debugging/inspection.
        """
        chunks_content = [c.page_content for c in chunks]
        chunks_metadata = [c.metadata for c in chunks]
        return chunks_content, chunks_metadata

    def get_used_input(self) -> dict:
        """
        Return configuration details of this agent for audit/debug purposes.
        """
        return {
            "retriever": "HybridRetriever",
            "llm": self.llm._identifying_params if self.llm else None
        }

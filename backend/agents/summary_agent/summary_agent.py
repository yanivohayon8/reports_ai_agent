import pathlib
import sys
import os
from typing import List, Dict, Optional, Union

backend_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, backend_path)

from backend.core.pdf_reader import read_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.agents.summary_agent.summary_prompts import (
    MAP_SUMMARY_PROMPT_CHAT_TEMPLATE,
    REDUCE_SUMMARY_PROMPT_CHAT_TEMPLATE,
    QUERY_BASED_SUMMARY_PROMPT_CHAT_TEMPLATE
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.documents import Document
from backend.agents.summary_agent.summary_chunker import SummaryChunker


class SummaryAgent:
    def __init__(self,
                 chunker: Optional[SummaryChunker] = None,
                 text_splitter: Optional[RecursiveCharacterTextSplitter] = None,
                 llm: Optional[BaseChatModel] = None):
        self.llm = llm
        self.chunker = chunker
        self.text_splitter = text_splitter or self._create_default_splitter()

    def _create_default_splitter(self):
        return RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    async def handle(self, query: str) -> Dict[str, str]:
        try:
            # If we have a chunker with indexed documents, use it
            if self.chunker and hasattr(self.chunker, 'faiss_indexer'):
                # Get relevant documents from the index
                relevant_docs = self.chunker.faiss_indexer.retrieve(query, num_documents=10)
                if relevant_docs:
                    context_text = "\n\n".join([doc.page_content for doc in relevant_docs])
                    result = self.summarize_with_query(query, context_text)
                    return {
                        "answer": result["answer"],
                        "agent": "SummaryAgent",
                        "reasoning": f"Generated summary from {len(relevant_docs)} relevant document chunks"
                    }
                else:
                    return {
                        "answer": "No relevant documents found in the index. Please ensure documents have been processed.",
                        "agent": "SummaryAgent",
                        "reasoning": "No indexed documents available"
                    }
            else:
                # Fallback to general summary
                base_context = "Please provide a general summary of the available information."
                result = self.summarize_with_query(query, base_context)
                return {
                    "answer": result["answer"],
                    "agent": "SummaryAgent",
                    "reasoning": "Generated summary from available context"
                }
        except Exception as e:
            return {"answer": f"Error generating summary: {str(e)}", "agent": "SummaryAgent"}

    def summarize_single_pdf(self, pdf_path: Union[str, pathlib.Path],
                             method: str = "map_reduce",
                             query: Optional[str] = None) -> Dict[str, str]:
        try:
            if self.chunker:
                docs: List[Document] = self.chunker.chunk(pdf_path)
                return self.summarize_from_docs(docs, method, query)
            else:
                docs = read_pdf(pdf_path, format="documents")
                return self.summarize_from_docs(docs, method, query)
        except Exception as e:
            return {"answer": f"Error reading PDF: {str(e)}"}

    def summarize_from_docs(self, docs: List[Document],
                            method: str = "map_reduce",
                            query: Optional[str] = None) -> Dict[str, str]:
        context_text = "\n".join([doc.page_content for doc in docs])
        metadata = [doc.metadata for doc in docs]

        if method == "map_reduce":
            result = self._summarize_map_reduce(context_text)
        elif method == "iterative":
            result = self._summarize_iterative_refinement(context_text)
        elif method == "query_based":
            result = self._summarize_query_based(context_text, query or "Provide a general summary")
        else:
            raise ValueError(f"Invalid summary method: {method}")

        result["chunks_metadata"] = metadata
        return result

    def summarize_with_query(self, query: str, context_text: str) -> Dict[str, str]:
        if not self.llm:
            return {"answer": "LLM not available."}

        chunks = self.text_splitter.split_text(context_text)
        if len(chunks) <= 1:
            chain = QUERY_BASED_SUMMARY_PROMPT_CHAT_TEMPLATE | self.llm
            response = chain.invoke({"query": query, "document": context_text})
            return {"answer": response.content}

        partial_summaries = []
        for chunk in chunks:
            chain = QUERY_BASED_SUMMARY_PROMPT_CHAT_TEMPLATE | self.llm
            response = chain.invoke({"query": query, "document": chunk})
            partial_summaries.append(response.content)

        combined_text = "\n\n".join(f"- {s}" for s in partial_summaries)
        reduce_chain = REDUCE_SUMMARY_PROMPT_CHAT_TEMPLATE | self.llm
        final_response = reduce_chain.invoke({"partial_summaries": combined_text})

        return {"answer": final_response.content, "chunks_summary": partial_summaries}

    # === Summarization methods ===
    def _summarize_map_reduce(self, text: str) -> Dict[str, str]:
        map_chain = MAP_SUMMARY_PROMPT_CHAT_TEMPLATE | self.llm
        reduce_chain = REDUCE_SUMMARY_PROMPT_CHAT_TEMPLATE | self.llm
        chunks = self.text_splitter.split_text(text)

        partial_summaries = []
        for chunk in chunks:
            resp = map_chain.invoke({"document": chunk})
            partial_summaries.append(resp.content)

        combined = "\n\n".join(partial_summaries)
        final = reduce_chain.invoke({"partial_summaries": combined})
        return {"answer": final.content, "chunks_summary": partial_summaries}

    def _summarize_iterative_refinement(self, text: str) -> Dict[str, str]:
        chunks = self.text_splitter.split_text(text)
        if not chunks:
            return {"answer": "No content."}

        init_chain = MAP_SUMMARY_PROMPT_CHAT_TEMPLATE | self.llm
        summary = init_chain.invoke({"document": chunks[0]}).content

        refine_chain = MAP_SUMMARY_PROMPT_CHAT_TEMPLATE | self.llm
        for chunk in chunks[1:]:
            resp = refine_chain.invoke({"document": chunk})
            summary = resp.content

        return {"answer": summary}

    def _summarize_query_based(self, text: str, query: str) -> Dict[str, str]:
        chain = QUERY_BASED_SUMMARY_PROMPT_CHAT_TEMPLATE | self.llm
        resp = chain.invoke({"query": query, "document": text})
        return {"answer": resp.content}

    def get_used_input(self):
        return {
            "text_splitter": {
                "chunk_size": self.text_splitter.chunk_size,
                "chunk_overlap": self.text_splitter.chunk_overlap,
            },
            "llm_model": self.llm._identifying_params if self.llm else None,
        }

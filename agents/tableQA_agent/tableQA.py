from typing import Any, Optional
import inspect
from langchain.prompts import PromptTemplate
from langchain_core.documents import Document
from retrieval.hybrid_retriever import HybridRetriever

# Provide a lightweight local LLMChain fallback to avoid external API usage in tests
try:
    from langchain.chains import LLMChain as _LC_LLMChain  # type: ignore
    LLMChain = _LC_LLMChain  # expose name for monkeypatch in tests
except Exception:  # pragma: no cover
    class LLMChain:  # minimal shim with the same name for monkeypatching
        def __init__(self, llm: Any = None, prompt: Any = None):
            self.llm = llm
            self.prompt = prompt
        def invoke(self, inputs: dict) -> dict:
            return {"text": ""}


class TableQAgent:
    def __init__(
        self,
        retriever: HybridRetriever,
        model_name: str = "gpt-4o-mini",
        llm_chain: Optional[Any] = None,
    ):
        """
        Agent specialized in answering questions on tabular data.
        Handles its own retrieval to keep Router simple.
        """
        self.retriever = retriever

        self.prompt = PromptTemplate(
            input_variables=["query", "table"],
            template=(
                "You are a Table Question Answering expert.\n"
                "The user provided a question and a table.\n"
                "Answer based only on the table content.\n\n"
                "Question: {query}\n\n"
                "Table:\n{table}\n\n"
                "Answer clearly and concisely."
            ),
        )
        # Construct chain without binding to a specific external LLM
        if llm_chain is not None:
            self.chain = llm_chain
        else:
            class _LocalChain:
                def __init__(self, prompt: Any):
                    self.prompt = prompt
                def invoke(self, inputs: dict) -> dict:
                    query = inputs.get("query", "")
                    table = inputs.get("table", "")
                    # Minimal deterministic answer without external calls
                    preview = (table[:120] + "...") if len(table) > 120 else table
                    return {"text": f"[Local TableQA] Q: {query}\nUsing table excerpt:\n{preview}"}
            self.chain = _LocalChain(self.prompt)

    def _extract_table(self, docs: list[Document]) -> str | None:
        """
        Find a document that looks like a table.
        Naive: checks for <table> or markdown style '|'.
        """
        for d in docs:
            if "<table>" in d.page_content or "|" in d.page_content:
                return d.page_content
        return None

    async def handle(self, query: str) -> str:
        """
        Retrieve relevant table, then answer question.
        """
        docs = self.retriever.retrieve(query, k_dense=3, k_sparse=3)
        table_text = self._extract_table(docs)

        if not table_text:
            return "No relevant table found in the documents."

        result = self.chain.invoke({"query": query, "table": table_text})
        if inspect.isawaitable(result):
            result = await result
        return result["text"]

"""TimelineSummarizer – high-level façade."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal

from langchain_openai import ChatOpenAI

from .splitter import split_into_docs
from .chains import build_map_reduce_chain, build_refine_chain

ChainType = Literal["map_reduce", "refine"]

@dataclass
class TimelineSummarizer:
    """Generate a timeline summary from text."""
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    method: ChainType = "map_reduce"
    chunk_size: int = 1_500
    chunk_overlap: int = 200
    token_based_split: bool = False
    _llm: ChatOpenAI = field(init=False, repr=False)

    def __post_init__(self):
        self._llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
        )
        self._chain = (
            build_map_reduce_chain(self._llm)
            if self.method == "map_reduce"
            else build_refine_chain(self._llm)
        )

    def summarize(self, text: str) -> str:
        """Produce a strict chronological timeline from *text*."""
        if not text.strip():
            return "⚠️ No readable text provided."

        docs = split_into_docs(
            text,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            token_based=self.token_based_split,
        )
        try:
            return self._chain.run(docs)
        except Exception as exc:  # noqa: BLE001
            return f"⚠️ Summarization failed: {exc}"

"""qna_prompts.py
Prompt templates for the QnA tool (English).
"""
from __future__ import annotations

from langchain.prompts import PromptTemplate

_GUIDELINES = (
    "- Answer ONLY from the provided context.\n"
    "- If the context is insufficient, say: 'I don't know'. Do NOT hallucinate.\n"
    "- Cite the exact snippet(s) you relied on (quote or paraphrase + location when available).\n"
    "- Keep answer language consistent with the user's question.\n"
    "- Be concise but complete: include all relevant facts from the context."
)

_QA_TMPL = """
You are a precise Question Answering assistant.

{guidelines}

---------------- CONTEXT START ----------------
{context}
----------------- CONTEXT END -----------------

Question: {question}

Final answer:
"""

def qa_prompt() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["context", "question"],
        template=_QA_TMPL,
        partial_variables={"guidelines": _GUIDELINES},
    )

_MULTIQUERY_TMPL = """
Rewrite the user question in {n} different ways to maximize recall in a retrieval system.
Return EXACTLY one variation per line, without numbering.

Original question:
{question}
"""

def multiquery_prompt() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["question", "n"],
        template=_MULTIQUERY_TMPL,
    )

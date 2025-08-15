from __future__ import annotations

import pathlib
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from typing import Literal
from timeline.io_utils import read_file
from timeline.summarizer import TimelineSummarizer
from qna.qna_core import build_qna_tool, QnATool
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor

load_dotenv()

class HaystackAgent:
    """LangChain agent with a custom prompt that classifies questions as NEEDLE or SUMMARY."""
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        summary_method: Literal["map_reduce","refine"] = "map_reduce",
        chunk_size: int = 1500,
        chunk_overlap: int = 200,
        token_based: bool = False,
        top_k: int = 4,
        score_threshold: float|None = None,
        persist: bool = False,
    ) -> None:
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        self.summariser = TimelineSummarizer(
            model_name=model_name,
            temperature=temperature,
            method=summary_method,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            token_based_split=token_based,
        )
        self._qna_kwargs = dict(
            model_name=model_name,
            temperature=temperature,
            top_k=top_k,
            score_threshold=score_threshold,
            persist_path=None if not persist else pathlib.Path("./qna_index.faiss"),
        )
        self.qna_tool: QnATool|None = None
        self._raw_text: str|None = None
        self._cached_summary: str|None = None
        self.agent_executor = None  # Will be set after tools are built

    def load_text(self, path: pathlib.Path) -> None:
        """Load raw text and build the QnA tool over it. Also build LangChain agent with custom prompt."""
        text = read_file(path)
        self._raw_text = text
        self.qna_tool = build_qna_tool(
            [text],
            token_based=self.summariser.token_based_split,
            chunk_size=self.summariser.chunk_size,
            chunk_overlap=self.summariser.chunk_overlap,
            top_k=self._qna_kwargs["top_k"],
            score_threshold=self._qna_kwargs["score_threshold"],
            persist_path=self._qna_kwargs["persist_path"],
            llm_model=self._qna_kwargs["model_name"],
            temperature=self._qna_kwargs["temperature"],
            return_sources=True,
        )
        self._cached_summary = None

        # Define LangChain-compatible tools
        def qna_tool_func(question: str) -> str:
            result = self.qna_tool.run(question)
            if isinstance(result, dict):
                answer = result.get("answer", "")
                sources = result.get("sources", [])
                if sources:
                    return f"{answer}\n\nSources:\n" + "\n".join(str(s) for s in sources)
                return answer
            return str(result)

        def summary_tool_func(_: str) -> str:
            if self._cached_summary is None:
                self._cached_summary = self.summariser.summarize(self._raw_text)
            return self._cached_summary

        qna_tool = Tool(
            name="QnA",
            func=qna_tool_func,
            description="Answer factual questions from the document."
        )
        summary_tool = Tool(
            name="Summary",
            func=summary_tool_func,
            description="Return a summary of the document."
        )

        # Use a custom agent prompt that includes all required variables
        custom_agent_prompt = PromptTemplate(
            input_variables=["question", "agent_scratchpad", "tools", "tool_names"],
            template="""
You are an expert assistant. Decide if the user question asks for a precise fact (NEEDLE) or a broad overview/synthesis (SUMMARY).

If NEEDLE, use the QnA tool to answer the question.
If SUMMARY, use the Summary tool to provide an overview.

You have access to the following tools:
{tools}

Tool names: {tool_names}

Question: {question}

{agent_scratchpad}
"""
        )

        # Build the agent with the custom prompt
        agent = create_react_agent(
            llm=self.llm,
            tools=[qna_tool, summary_tool],
            prompt=custom_agent_prompt
        )
        self.agent_executor = AgentExecutor(agent=agent, tools=[qna_tool, summary_tool], verbose=False)

    def answer(self, question: str) -> str:
        if self.agent_executor is None:
            raise RuntimeError("Must call load_text() before asking questions.")
        result = self.agent_executor.invoke({"question": question})
        # Try to extract the main output (usually 'output' or the first value)
        if isinstance(result, dict):
            if "output" in result:
                return result["output"]
            elif len(result) > 0:
                return next(iter(result.values()))
            else:
                return ""
        return str(result)

import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from core.api_utils import get_llm_langchain_openai
from core.text_splitter import get_text_splitter
from agents.summary_agent.summary import SummaryAgent


def test_insurance_report_map_reduce():
    text_splitter = get_text_splitter()
    llm = get_llm_langchain_openai(model="gpt-4o-mini")
    summary_agent = SummaryAgent(text_splitter,llm)
    summary = summary_agent.summarize_single_pdf(pathlib.Path("data/report.pdf"), "map_reduce")
    assert summary is not None
    assert len(summary) > 0


def test_insurance_report_iterative():
    text_splitter = get_text_splitter()
    llm = get_llm_langchain_openai(model="gpt-4o-mini")
    summary_agent = SummaryAgent(text_splitter,llm)
    summary = summary_agent.summarize_single_pdf(pathlib.Path("data/report.pdf"), "iterative")

    assert summary is not None
    assert len(summary) > 0



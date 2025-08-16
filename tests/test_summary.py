from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.api_utils import get_llm_langchain_openai
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from summary_agent.summary import SummaryAgent


def test_insurance_report_map_reduce():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50)
    llm = get_llm_langchain_openai(model="gpt-4o-mini")
    summary_agent = SummaryAgent(text_splitter,llm)
    summary = summary_agent.summarize_single_pdf(pathlib.Path("pdfs/report.pdf"), "map_reduce")
    assert summary is not None
    assert len(summary) > 0


def test_insurance_report_iterative():
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50)
    llm = get_llm_langchain_openai(model="gpt-4o-mini")
    summary_agent = SummaryAgent(text_splitter,llm)
    summary = summary_agent.summarize_single_pdf(pathlib.Path("pdfs/report.pdf"), "iterative")

    assert summary is not None
    assert len(summary) > 0



import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).parent.parent))

from summary_agent.summary import SummaryAgent


def test_insurance_report():
    summary_agent = SummaryAgent()
    summary = summary_agent.summarize_single_pdf(pathlib.Path("pdfs/report.pdf"), "iterative")
    assert summary is not None


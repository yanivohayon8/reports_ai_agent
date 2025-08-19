import pytest
from pathlib import Path
import sys
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from core import pdf_reader


def test_event_report_2():
    pdf_path = Path("tests/data/event_report_2_extended.pdf")
    pages = pdf_reader.read_pdf(pdf_path)
    assert pages is not None
    assert len(pages) == 2
    assert len(pages[0].metadata["tables"]) == 0
    assert len(pages[1].metadata["tables"]) == 1
    assert isinstance(pages[1].metadata["tables"][0],pd.DataFrame)

def test_event_report_2_as_text():
    pdf_path = Path("tests/data/event_report_2_extended.pdf")
    assert pdf_path.exists()

    text = pdf_reader.read_pdf(pdf_path,format="text")
    assert text is not None
    assert len(text) > 0
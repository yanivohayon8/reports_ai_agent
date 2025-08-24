import os
import pytest
from pathlib import Path
import sys
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

# Skip if llama_parse is not installed or API key missing
pytest.importorskip("llama_parse")
if not os.environ.get("LLAMA_CLOUD_API_KEY"):
    pytest.skip("LLAMA_CLOUD_API_KEY not set; skipping PDF parsing tests.", allow_module_level=True)

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

def test_event_report_2_as_text_no_split_by_page():
    pdf_path = Path("tests/data/event_report_2_extended.pdf")
    text = pdf_reader.read_pdf(pdf_path,format="text",split_by_page=False)

    assert text is not None


def test_extract_tables():
    pdf_path = Path("tests/data/client2_report2_tourAndCarePolicy.pdf")
    assert pdf_path.exists()

    tables = pdf_reader.extract_tables(pdf_path)

    assert len(tables) == 7
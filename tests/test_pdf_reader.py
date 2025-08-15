import pathlib
import pytest
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pdf_reader import read_pdf

@pytest.fixture
def pdf_path():
    return pathlib.Path("tests/data/report.pdf")

def test_read_pdf(pdf_path):
    text = read_pdf(pdf_path)
    assert text is not None
    assert len(text) > 0
    print(text[:100])
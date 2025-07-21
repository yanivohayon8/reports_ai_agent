# tests/test_utils.py

import sys
sys.path.append("./")
import os
import pytest
from src.utils import read_pdf_text, get_openai_model
from langchain import OpenAI
import builtins

def test_read_pdf_text():
    pdf_path = os.path.join('data', 'report.pdf')
    text = read_pdf_text(pdf_path)
    assert isinstance(text, str)
    assert len(text) > 0, 'PDF text should not be empty.'

def test_get_openai_model(monkeypatch):
    # Mock getpass.getpass to avoid interactive prompt
    monkeypatch.setattr('getpass.getpass', lambda prompt: 'sk-test-key')
    # Remove API key from environment if present
    if 'OPENAI_API_KEY' in os.environ:
        del os.environ['OPENAI_API_KEY']
    model = get_openai_model('gpt-3.5-turbo')
    assert isinstance(model, OpenAI) 
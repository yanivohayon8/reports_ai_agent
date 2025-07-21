# utils.py
# Utility functions for the AI agent project

import PyPDF2
import getpass
import os
from langchain import OpenAI

def _set_openai_api_key():
    if "OPENAI_API_KEY" not in os.environ:
        os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter your OpenAI API key: ")

def get_openai_model(model_name="gpt-4o-mini"):
    _set_openai_api_key()
    return OpenAI(model=model_name)

def read_pdf_text(pdf_path):
    """
    Reads a PDF file and returns its text content as a string.
    """
    text = ""
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            text += page.extract_text() or ""
    return text 

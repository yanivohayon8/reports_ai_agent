import os
from agents.needle_agent.evaluator import DatasetCreator
from core.pdf_reader import read_pdf
from pathlib import Path

def test_dataset_create_chunk():
    dataset_creator = DatasetCreator.from_gpt()    
    pdf_path = Path(os.path.join("tests","data","report.pdf"))
    text = read_pdf(pdf_path, format="text")
    chunk_size = 400
    chunk_overlap = 100
    chunk = dataset_creator._create_random_chunk(text, chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    assert isinstance(chunk, str)

    response_json = dataset_creator._generate_answer_and_question(chunk)
    assert isinstance(response_json, dict)
    assert "question" in response_json
    assert "answer" in response_json



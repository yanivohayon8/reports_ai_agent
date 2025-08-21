import os
from agents.needle_agent.evaluator import DatasetCreator
from core.pdf_reader import read_pdf
from pathlib import Path

def test_dataset_create_chunk():
    output_directory = "tests/data/dataset"
    dataset_creator = DatasetCreator(output_directory)    
    pdf_path = Path(os.path.join("tests","data","report.pdf"))
    text = read_pdf(pdf_path, format="text")
    chunk_size = 400
    chunk_overlap = 100
    chunk = dataset_creator._create_random_chunk(text, chunk_size=chunk_size,chunk_overlap=chunk_overlap)
    
    assert isinstance(chunk, str)



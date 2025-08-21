import os
from agents.needle_agent.evaluator import DatasetCreator
from core.pdf_reader import read_pdf
from pathlib import Path
from core.api_utils import get_llm_langchain_openai


def test_dataset_create_chunk():
    llm = get_llm_langchain_openai(model="gpt-4o-mini")
    dataset_creator = DatasetCreator(llm)    
    pdf_path = Path(os.path.join("tests","data","report.pdf"))
    text = read_pdf(pdf_path, format="text")
    chunk_size = 400
    chunk_overlap = 100
    chunks = dataset_creator._create_chunks_by_size(text, 
                                                   chunk_size=chunk_size,
                                                   chunk_overlap=chunk_overlap)
    assert isinstance(chunks, list)
    assert len(chunks) > 0
    assert isinstance(chunks[0], str)

    response_json = dataset_creator._generate_answer_and_question(chunks[0])
    assert isinstance(response_json, dict)
    assert "question" in response_json
    assert "answer" in response_json

def test_create_chunks():
    llm = get_llm_langchain_openai(model="gpt-4o-mini")
    dataset_creator = DatasetCreator(llm)    
    pdf_path = Path(os.path.join("tests","data","report.pdf"))
    text = read_pdf(pdf_path, format="text")

    questions_and_answers = dataset_creator.create_dataset_from_text(text, n_chunks=2)
    assert isinstance(questions_and_answers, list)
    assert len(questions_and_answers) > 0
    assert isinstance(questions_and_answers[0], dict)
    assert "question" in questions_and_answers[0]
    assert "answer" in questions_and_answers[0]








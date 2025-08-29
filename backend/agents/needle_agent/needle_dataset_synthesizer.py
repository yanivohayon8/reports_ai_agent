import sys
import os

# Add the backend directory to Python path
backend_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, backend_path)

from core.pdf_reader import read_pdf
from core.text_splitter import get_text_splitter
import random
from langchain_core.language_models import BaseChatModel
from agents.needle_agent.needle_prompts import evaluation_Qna_template
import json
from pathlib import Path

class DatasetSynthesizer:
    def __init__(self, llm:BaseChatModel, 
                 min_chunk_size: int, max_chunk_size: int,
                 min_chunk_overlap: int, max_chunk_overlap: int):
        self.llm = llm
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_chunk_overlap = min_chunk_overlap
        self.max_chunk_overlap = max_chunk_overlap

    def create_dataset(self, pdf_path: Path, n_chunks: int=10):
        text = read_pdf(pdf_path, format="text")
        questions_and_answers = self.create_dataset_from_text(text, n_chunks=n_chunks)

        for question_and_answer in questions_and_answers:
            question_and_answer["file_path"] = str(pdf_path)

        return questions_and_answers

    def create_dataset_from_text(self, text: str, n_chunks: int=10):
        chunks = self._create_chunks(text)
        random_chunks = self._select_random_chunk(chunks, n_chunks=n_chunks)
        questions_and_answers = self._generate_questions_and_answers(random_chunks)

        return questions_and_answers

    def _create_chunks(self, text: str):
        chunk_size = random.randint(self.min_chunk_size, self.max_chunk_size)
        chunk_overlap = random.randint(self.min_chunk_overlap, self.max_chunk_overlap)
        chunks = self._create_chunks_by_size(text, chunk_size, chunk_overlap)
        return chunks

    def _create_chunks_by_size(self,text: str, chunk_size: int, chunk_overlap: int):
        text_splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_text(text)
        return chunks
    
    def _select_random_chunk(self,chunks: list[str], n_chunks: int)->list[str]:
        random_chunks = chunks if n_chunks >= len(chunks) else random.sample(chunks, n_chunks)
        return random_chunks
    
    def _generate_questions_and_answers(self,chunks: list[str])->list[dict]:
        questions_and_answers = []
        for chunk in chunks:
            question_and_answer = self._generate_answer_and_question(chunk)
            question_and_answer["reference_chunk"] = chunk
            questions_and_answers.append(question_and_answer)
        return questions_and_answers

    def _generate_answer_and_question(self,chunk: str)->dict:
        prompt = evaluation_Qna_template.invoke({"chunk":chunk})
        response = self.llm.invoke(prompt)
        question_and_answer = json.loads(response.content)

        return question_and_answer


def save_needle_dataset(questions_and_answers: list[dict], pdf_path: Path, output_dir: Path) -> int:
    """
    Save questions and answers dataset to a JSONL file.
    
    Args:
        questions_and_answers: List of question-answer dictionaries
        pdf_path: Path to the source PDF file
        output_dir: Directory to save the output file
    
    Returns:
        int: Number of question-answer pairs saved
    """
    # Create filename based on PDF name
    pdf_name = pdf_path.stem  # Remove .pdf extension
    output_file = output_dir / f"{pdf_name}.jsonl"
    
    print(f"Processing: {pdf_path.name}")
    print(f"Saving dataset to: {output_file}")
    
    with open(output_file, "w") as f:
        for question_and_answer in questions_and_answers:
            f.write(json.dumps(question_and_answer) + "\n")
    
    print(f"Dataset saved successfully with {len(questions_and_answers)} question-answer pairs")
    return len(questions_and_answers)


def load_needle_dataset(dataset_path: Path) -> list[dict]:
    """
    Load questions and answers dataset from a JSONL file.
    
    Args:
        dataset_path: Path to the JSONL dataset file
    
    Returns:
        list[dict]: List of question-answer dictionaries
    """
    questions_and_answers = []
    
    with open(dataset_path, "r") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                questions_and_answers.append(json.loads(line.strip()))
    
    print(f"Loaded dataset from {dataset_path} with {len(questions_and_answers)} question-answer pairs")
    return questions_and_answers


    

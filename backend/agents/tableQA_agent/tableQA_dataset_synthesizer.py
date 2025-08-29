import sys
import os
import random
import json
from pathlib import Path
from typing import List

# Add the backend directory to Python path
backend_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, backend_path)

from backend.core.pdf_reader import read_pdf
from backend.core.text_splitter import get_text_splitter
from langchain_core.language_models import BaseChatModel
from backend.agents.needle_agent.needle_prompts import evaluation_Qna_template


class TableQADatasetSynthesizer:
    def __init__(
        self,
        llm: BaseChatModel,
        min_chunk_size: int,
        max_chunk_size: int,
        min_chunk_overlap: int,
        max_chunk_overlap: int,
    ) -> None:
        self.llm = llm
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_chunk_overlap = min_chunk_overlap
        self.max_chunk_overlap = max_chunk_overlap

    def create_dataset(self, pdf_path: Path, n_chunks: int = 10) -> List[dict]:
        text = read_pdf(pdf_path, format="text")
        rows = self.create_dataset_from_text(text, n_chunks=n_chunks)
        for row in rows:
            row["file_path"] = str(pdf_path)
        return rows

    def create_dataset_from_text(self, text: str, n_chunks: int = 10) -> List[dict]:
        chunks = self._create_chunks(text)
        random_chunks = self._select_random_chunk(chunks, n_chunks=n_chunks)
        rows = self._generate_questions_and_answers(random_chunks)
        return rows

    def _create_chunks(self, text: str) -> List[str]:
        chunk_size = random.randint(self.min_chunk_size, self.max_chunk_size)
        chunk_overlap = random.randint(self.min_chunk_overlap, self.max_chunk_overlap)
        return self._create_chunks_by_size(text, chunk_size, chunk_overlap)

    def _create_chunks_by_size(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        return splitter.split_text(text)

    def _select_random_chunk(self, chunks: List[str], n_chunks: int) -> List[str]:
        return chunks if n_chunks >= len(chunks) else random.sample(chunks, n_chunks)

    def _generate_questions_and_answers(self, chunks: List[str]) -> List[dict]:
        rows: List[dict] = []
        for chunk in chunks:
            qa = self._generate_answer_and_question(chunk)
            qa["context_text"] = chunk
            rows.append(qa)
        return rows

    def _generate_answer_and_question(self, chunk: str) -> dict:
        prompt = evaluation_Qna_template.invoke({"chunk": chunk})
        response = self.llm.invoke(prompt)
        return json.loads(response.content)


def save_tableQA_dataset(rows: List[dict], pdf_path: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_name = pdf_path.stem
    output_file = output_dir / f"{pdf_name}.jsonl"

    with open(output_file, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return len(rows)


def load_tableQA_dataset(dataset_path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

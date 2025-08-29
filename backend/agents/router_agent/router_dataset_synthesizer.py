import os
import sys
import json
import random
from pathlib import Path
from typing import List

# Ensure backend is importable
backend_path = os.path.join(os.path.dirname(__file__), '..', '..')
if backend_path not in sys.path:
    sys.path.insert(0, backend_path)

from backend.core.pdf_reader import read_pdf
from backend.core.text_splitter import get_text_splitter
from backend.core.api_utils import get_llm_langchain_openai
from backend.agents.router_agent.router import classify_query


def _make_questions_from_chunk(chunk: str, llm_model: str | None) -> List[str]:
    """Generate 1-3 questions for a given text chunk.
    Uses LLM if available; falls back to simple template otherwise.
    """
    if llm_model:
        try:
            llm = get_llm_langchain_openai(model=llm_model)
            prompt = (
                "Read the following text and write 3 diverse user questions: "
                "1) a direct fact lookup (needle), 2) a general summary question (summary), "
                "3) a table-related question if there are tables, otherwise another summary.\n\n"
                f"Text:\n{chunk}\n\nReturn as a numbered list only."
            )
            resp = llm.invoke(prompt)
            lines = [l.strip("- ") for l in resp.content.splitlines() if l.strip()]
            if not lines:
                return [f"Summarize: {chunk[:120]}..."]
            # take first up to 3
            return lines[:3]
        except Exception:
            pass
    # Fallback
    return [
        f"What is a key fact mentioned?",
        f"Summarize the main idea.",
        f"Are there any tables and what do they show?",
    ]


def create_router_dataset_from_text(text: str, *, chunk_size: int, chunk_overlap: int, n_chunks: int) -> List[dict]:
    splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    sampled = chunks if n_chunks >= len(chunks) else random.sample(chunks, n_chunks)

    rows: List[dict] = []
    for ch in sampled:
        questions = _make_questions_from_chunk(ch, llm_model=None)
        for q in questions:
            cls = classify_query(q)
            rows.append({
                "question": q,
                "expected_type": cls.type,
                "reasoning": cls.reasoning,
                "context_text": ch,
            })
    return rows


def create_router_dataset_from_pdf(pdf_path: Path, *, chunk_size: int, chunk_overlap: int, n_chunks: int) -> List[dict]:
    text = read_pdf(pdf_path, format="text")
    rows = create_router_dataset_from_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap, n_chunks=n_chunks)
    for r in rows:
        r["file_path"] = str(pdf_path)
    return rows


def save_router_dataset(rows: List[dict], output_dir: Path, base_name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out = output_dir / f"{base_name}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return out


def load_router_dataset(path: Path) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows



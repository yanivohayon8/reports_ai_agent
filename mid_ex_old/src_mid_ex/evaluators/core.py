# evaluators/core.py
"""Shared utilities for evaluator modules (I/O, dataset building, constants)."""
from __future__ import annotations

import csv
import json
import pathlib
from typing import List, Dict

import PyPDF2
import pandas as pd
from datasets import Dataset
from ragas.metrics import context_precision, context_recall, faithfulness, answer_correctness

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------
RAGAS_METRICS_MAP: Dict[str, object] = {
    "context_precision": context_precision,
    "context_recall": context_recall,
    "faithfulness": faithfulness,
    "answer_correctness": answer_correctness,
}

# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def read_pdf(path: pathlib.Path) -> str:
    """Extract plain text from a PDF (simple PyPDF2 extraction)."""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def read_any(path: pathlib.Path) -> str:
    """Read txt/markdown/pdf and fallback to common encodings."""
    if path.suffix.lower() == ".pdf":
        return read_pdf(path)
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"Cannot decode {path}")

# ---------------------------------------------------------------------------
# Question helpers
# ---------------------------------------------------------------------------

def load_questions(path: pathlib.Path) -> List[str]:
    """Load questions from .txt, .csv (column `question`), or .jsonl."""
    if path.suffix.lower() in {".txt", ".md"}:
        return [ln.strip() for ln in read_any(path).splitlines() if ln.strip()]
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        if "question" not in df.columns:
            raise ValueError("CSV must include a 'question' column")
        return df["question"].astype(str).tolist()
    if path.suffix.lower() in {".jsonl", ".json"}:
        return [json.loads(l).get("question", "").strip() for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    raise ValueError("Unsupported questions file – expected .txt / .csv / .jsonl")

# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_rows(path: pathlib.Path, rows: List[dict]) -> None:
    """Dump list‑of‑dicts to CSV or JSONL, decided by file suffix."""
    if not rows:
        return
    if path.suffix.lower() == ".csv":
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader(); w.writerows(rows)
    elif path.suffix.lower() in {".jsonl", ".json"}:
        path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")
    else:
        raise ValueError("Output must end with .csv / .jsonl / .json")

# ---------------------------------------------------------------------------
# Dataset builder – compatible with RAGAS >=0.1.0
# ----------------------------------------------------------------------------

def build_ragas_dataset(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truth: str,
) -> Dataset:
    """Build a Dataset that works with both old and new RAGAS versions.

    Columns added:
    * ``reference``   – single string (new schema ≥0.1.0)
    * ``references``  – list[str]   (beta schema)
    * ``ground_truths`` – list[list[str]] (legacy ≤0.0.17)
    """
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "reference": [ground_truth] * len(questions),
        "references": [[ground_truth]] * len(questions),
        "ground_truths": [[ground_truth]] * len(questions),
    }
    return Dataset.from_dict(data)

def build_ragas_dataset_old_schema(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    ground_truth: str,
) -> Dataset:
    """Build a Dataset that works with old RAGAS versions.

    Columns added:
    * ``reference``   – single string (new schema ≥0.1.0)
    * ``references``  – list[str]   (beta schema)
    * ``ground_truths`` – list[list[str]] (legacy ≤0.0.17)
    """
    data = {    
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        # Legacy (<=0.0.17) field name
        "ground_truths": [[ground_truth]] * len(questions),
        # Newer versions use `references` (plural) …
        "references": [[ground_truth]] * len(questions),
        # … while 0.1.0‑rc sometimes expects singular `reference`
        "reference": [[ground_truth]] * len(questions)
    }
    return Dataset.from_dict(data)
"""qna_eval.py
Batch-evaluate the QnA tool with RAGAS.
- Supports GT (ground truth) per question from a CSV.
- Falls back to a single GT text if a path isn't provided.
"""
from __future__ import annotations

import pathlib
from typing import List, Dict, Any, Union

import pandas as pd
from datasets import Dataset
from ragas import evaluate as ragas_evaluate
from dotenv import load_dotenv

from evaluators.core import (
    RAGAS_METRICS_MAP,
    load_questions,
    save_rows,
)

# Import the new builder
from tools.tools_project.qna.qna_core import build_qna_tool 

load_dotenv()


class QnaEvaluator:
    """Answer many questions and score them with chosen RAGAS metrics."""

    def __init__(
        self,
        doc_path: pathlib.Path,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        top_k: int = 12,
    ) -> None:
        self.top_k = top_k
        text = self._read_file(doc_path)

        # Build QnA tool with sources (needed for context metrics)
        self.qna = build_qna_tool(
            [text],
            llm_model=model_name,
            temperature=temperature,
            # ------- retrieval -------
            top_k=30,
            fetch_k=120,
            lambda_mult=0.35,
            score_threshold=None,
            use_multiquery=True,
            n_queries=4,
            # ------- chunking -------
            token_based=True,
            chunk_size=1400,
            chunk_overlap=200,
            # ------- misc -------
            return_sources=False,
        )
        self.retriever = self.qna.retriever

    # ------------------------------------------------------------------
    def run_batch(
        self,
        questions_file: pathlib.Path,
        answers_out: pathlib.Path,
        gt_source: Union[str, pathlib.Path],
        metrics: List[str] | None = None,
    ) -> Dict[str, float] | Any:
        """
        gt_source: either
          * Path to CSV with columns: question, ground_truth
          * a single string (full document) used for all questions
        """
        metrics = metrics or ["context_precision", "context_recall", "faithfulness", "answer_correctness"]
        metric_fns = [RAGAS_METRICS_MAP[m] for m in metrics]

        # ---------------- Execute QnA ----------------
        questions = load_questions(questions_file)
        rows, ctxs, answers = [], [], []
        for q in questions:
            res = self.qna.run(q)
            
            if isinstance(res, dict):
                answer=res["answer"]
            else:
                answer = res # assuming str
            
            raw_docs = self.retriever.get_relevant_documents(q)
            retrieved = [d.page_content for d in raw_docs]

            rows.append({"question": q, "answer": answer})
            answers.append(answer)
            ctxs.append(retrieved)

            print(f"Question: {q}")
            print(f"Answer: {answer}")
            print(f"Contexts: {retrieved}")
            print("-" * 100)

        save_rows(answers_out, rows)

        # ---------------- Ground Truth ----------------
        gts = self._load_gt(gt_source, questions)

        # ---------------- Evaluate ------------------
        dataset: Dataset = self._build_ragas_dataset(questions, answers, ctxs, gts)
        ragas_res = ragas_evaluate(dataset, metrics=metric_fns, raise_exceptions=False)
        return ragas_res

    # ------------------- helpers -------------------
    @staticmethod
    def _read_file(path: pathlib.Path) -> str:
        ext = path.suffix.lower()
        if ext == ".pdf":
            from pypdf import PdfReader
            return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages)
        return path.read_text(encoding="utf-8")

    @staticmethod
    def _load_gt(gt_source: Union[str, pathlib.Path], questions: List[str]) -> List[str]:
        if isinstance(gt_source, pathlib.Path) and gt_source.exists():
            df = pd.read_csv(gt_source)
            # Expect columns: question, ground_truth
            if "question" not in df or "ground_truth" not in df:
                raise ValueError("GT CSV must have 'question' and 'ground_truth' columns")
            gt_map = dict(zip(df["question"], df["ground_truth"]))
            return [gt_map.get(q, "") for q in questions]
        # Single text used for all
        return [str(gt_source)] * len(questions)

    @staticmethod
    def _build_ragas_dataset(
        questions: List[str],
        answers: List[str],
        contexts: List[List[str]],
        ground_truths: List[str],
    ) -> Dataset:
        return Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )
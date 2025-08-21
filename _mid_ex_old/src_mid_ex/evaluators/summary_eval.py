# evaluators/summary_eval.py
"""Summary generation + evaluation wrapper.
This module provides `SummaryEvaluator`, a convenience class that:
1. Builds a `TimelineSummarizer` on top of your project’s summarizer.
2. Generates a summary for an entire document.
3. Evaluates that summary with selected metrics:
* **faithfulness**  – RAGAS metric comparing summary claims to source text.
* **rougeL / bertscore** – optional overlap / semantic‑similarity metrics that
    require a *reference* (gold) summary.   

The import block is flexible so the code runs regardless of where your
`summarizer.py` and `qna_core.py` live:

* `tools_project.timeline.summarizer.TimelineSummarizer`  (your current tree)
* `tools.summarizer.TimelineSummarizer`                    (if you move files)
* local `summarizer.py`                                    (fallback)
"""
from __future__ import annotations

import pathlib
from typing import List, Dict

from datasets import Dataset
from evaluate import load as hf_load
from ragas import evaluate as ragas_evaluate
from ragas.metrics import faithfulness

from evaluators.core import read_any

# ---------------------------------------------------------------------------
# Flexible imports – summarizer & document‑splitter
# ---------------------------------------------------------------------------
from tools.tools_project.timeline.summarizer import TimelineSummarizer , split_into_docs

# ---------------------------------------------------------------------------
class SummaryEvaluator:
    """Generate a document‑level summary and compute chosen metrics.

    Parameters
    ----------
    doc_path : pathlib.Path
        Path to the source document (txt or pdf).
    model_name : str, default "gpt-4o-mini"
        Name of the LLM used by the underlying summarizer.
    temperature : float, default 0.0
        Temperature for the LLM.
    """

    DEFAULT_METRICS = ["faithfulness", "rougeL"]

    def __init__(
        self,
        doc_path: pathlib.Path,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ) -> None:
        # Load full document as plain text
        self.raw_text = read_any(doc_path)
        # Chunk the text using the same splitter as QnA
        self.docs = split_into_docs(self.raw_text)
        # Create project‑specific summarizer
        self.summarizer = TimelineSummarizer(
            model_name=model_name,
            temperature=temperature,
            method="map_reduce",  # change to "refine" if preferred
        )

    # ------------------------------------------------------------------
    def generate(self) -> str:
        """Return a summary string for the loaded document."""
        return self.summarizer.summarize(self.raw_text)

    # ------------------------------------------------------------------
    def evaluate(
        self,
        summary: str,
        ground_text: str,
        ref_summary_path: pathlib.Path | None = None,
        metrics: List[str] | None = None,
    ) -> Dict[str, float]:
        """Compute selected metrics for a given summary.

        Parameters
        ----------
        summary : str
            The summary to evaluate.
        ground_text : str
            Full source document text (used for faithfulness).
        ref_summary_path : pathlib.Path | None, optional
            Path to a *reference* summary. Required for ROUGE/BERTScore.
        metrics : list[str] | None, optional
            Metrics to calculate. Defaults to :pyattr:`DEFAULT_METRICS`.
        """
        metrics = metrics or self.DEFAULT_METRICS
        results: Dict[str, float] = {}

        # --- Faithfulness (RAGAS) ---
        if "faithfulness" in metrics:
            ds = Dataset.from_dict({
                "question": ["SUMMARY"],
                "answer": [summary],
                "contexts": [[d.page_content for d in self.docs]],
                "ground_truths": [[ground_text]],
            })
            score = ragas_evaluate(ds, metrics=[faithfulness], raise_exceptions=False)
            faith = score["faithfulness"]
            if isinstance(faith, list):
                faith = faith[0] if faith else None
            results["faithfulness"] = float(faith) if faith is not None else None

        # --- ROUGE / BERTScore ---
        if ref_summary_path and ref_summary_path.exists():
            ref_sum = read_any(ref_summary_path)
            if "rougeL" in metrics:
                rouge = hf_load("rouge")
                r = rouge.compute(
                    predictions=[summary],
                    references=[ref_sum],
                    rouge_types=["rougeL"],
                )
                results["rougeL"] = float(r["rougeL"])
            if "bertscore" in metrics:
                b = hf_load("bertscore").compute(
                    predictions=[summary],
                    references=[ref_sum],
                    lang="en",
                )
                results["bertscore"] = float(b["f1"][0])
        elif any(m in metrics for m in ("rougeL", "bertscore")):
            print("⚠️  Skipping ROUGE/BERTScore – provide --summary-ref for these metrics.")

        return results

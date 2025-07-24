"""
summary_grid_search.py
Grid search for your summarization pipeline using SummaryEvaluator + RAGAS/ROUGE/BERTScore.

Usage:
    python summary_grid_search.py \
        --doc data/report.pdf \
        --ref data/reference_summary.txt \
        --out summary_grid.csv

- If you don't have a reference summary, omit --ref and metrics rougeL/bertscore will be skipped.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

import pandas as pd
from ragas.metrics import faithfulness, answer_correctness
from ragas import evaluate as ragas_evaluate
from datasets import Dataset
from evaluate import load as hf_load

from evaluators.core import read_any  # adjust path if needed
from evaluators.summary_eval import SummaryEvaluator  # your existing class
from dotenv import load_dotenv

load_dotenv()


# -------------------------- Config object -------------------------- #
@dataclass
class RunConfig:
    method: str                 # "map_reduce" | "refine"
    chunk_size: int
    chunk_overlap: int
    temperature: float
    model_name: str
    n_passes: int

    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


# ------------------------- Param grid ------------------------------ #
def build_param_grid() -> List[RunConfig]:
    base_models = ["gpt-4o-mini"]     # add others if you want
    methods = ["map_reduce", "refine"]
    chunk_sizes = [800, 1200, 1600]
    overlaps = [120, 200, 300]
    temps = [0.0, 0.2]
    n_passes = [1, 2, 3]

    grid: List[RunConfig] = []
    for m, cs, ov, t, model, n_passes in itertools.product(methods, chunk_sizes, overlaps, temps, base_models, n_passes):
        grid.append(RunConfig(method=m, chunk_size=cs, chunk_overlap=ov, temperature=t, model_name=model, n_passes=n_passes))
    return grid


# ------------------------- Metrics helpers ------------------------- #
def eval_faithfulness(summary: str, full_text_chunks: List[str]) -> float:
    ds = Dataset.from_dict({
        "question": ["SUMMARY"],
        "answer": [summary],
        "contexts": [full_text_chunks],
        "ground_truths": [full_text_chunks],  # full text as “truth”
    })
    res = ragas_evaluate(ds, metrics=[faithfulness], raise_exceptions=False)
    val = res["faithfulness"]
    if hasattr(val, "value"):
        return float(val.value)
    if isinstance(val, list) and val:
        return float(val[0])
    return float(val) if isinstance(val, (int, float)) else 0.0


def eval_overlap(summary: str, ref_summary: str, do_rouge: bool, do_bert: bool) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    if do_rouge:
        rouge = hf_load("rouge")
        r = rouge.compute(predictions=[summary], references=[ref_summary], rouge_types=["rougeL"])
        scores["rougeL"] = float(r["rougeL"])
    if do_bert:
        bert = hf_load("bertscore").compute(predictions=[summary], references=[ref_summary], lang="en")
        scores["bertscore"] = float(bert["f1"][0])
    return scores


def eval_answer_correctness(summary: str, ref_summary: str) -> float:
    ds = Dataset.from_dict({
        "question": ["SUMMARY"],
        "answer": [summary],
        "contexts": [[ref_summary]],
        "reference": [ref_summary],  # Changed from "ground_truths" to "reference"
    })
    res = ragas_evaluate(ds, metrics=[answer_correctness], raise_exceptions=False)
    val = res["answer_correctness"]
    if hasattr(val, "value"):
        return float(val.value)
    if isinstance(val, list) and val:
        return float(val[0])
    return float(val) if isinstance(val, (int, float)) else 0.0


# ------------------------- Single run ------------------------------ #
def run_single(cfg: RunConfig, doc_path: pathlib.Path, full_text: str,
               ref_text: Optional[str]) -> Dict[str, Any]:
    # Build evaluator with chosen params
    seval = SummaryEvaluator(doc_path, model_name=cfg.model_name, temperature=cfg.temperature)
    # Override its chunk params & method if needed
    # (assuming TimelineSummarizer takes method & you can change split params globally;
    # otherwise modify your summarizer to accept them)
    seval.summarizer.method = cfg.method
    # re-split with new params
    from tools.tools_project.timeline.summarizer import split_into_docs  # adjust import
    seval.docs = split_into_docs(full_text, chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)

    summary = seval.summarizer.summarize(full_text)

    # faithfulness
    faith = eval_faithfulness(summary, [d.page_content for d in seval.docs])

    # rouge / bertscore / answer_correctness
    overlap_scores = {}
    answer_corr = None
    if ref_text:
        overlap_scores = eval_overlap(
            summary, ref_text,
            do_rouge=True,
            do_bert=True
        )
        answer_corr = eval_answer_correctness(summary, ref_text)

    out = {
        "faithfulness": faith,
        "rougeL": overlap_scores.get("rougeL"),
        "bertscore": overlap_scores.get("bertscore"),
        "answer_correctness": answer_corr,
        **asdict(cfg),
        "summary_len": len(summary),
    }
    return out


# ---------------------------- Main -------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Grid search for summary hyper-params")
    ap.add_argument("--doc", type=pathlib.Path, required=True, help="Source document (txt/pdf)")
    ap.add_argument("--ref", type=pathlib.Path, help="Reference summary for ROUGE/BERTScore")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("summary_grid.csv"))
    ap.add_argument("--max-runs", type=int, default=None, help="Limit configs for quick debug")
    args = ap.parse_args()

    full_text = read_any(args.doc)
    ref_text = read_any(args.ref) if args.ref and args.ref.exists() else None

    grid = build_param_grid()
    if args.max_runs:
        grid = grid[: args.max_runs]

    # resume logic
    results: List[Dict[str, Any]] = []
    done_keys = set()
    if args.out.exists():
        try:
            old = pd.read_csv(args.out)
            if "method" in old:
                for _, row in old.iterrows():
                    key = json.dumps({
                        "method": row.get("method"),
                        "chunk_size": row.get("chunk_size"),
                        "chunk_overlap": row.get("chunk_overlap"),
                        "temperature": row.get("temperature"),
                        "model_name": row.get("model_name"),
                    }, sort_keys=True)
                    done_keys.add(key)
            results = old.to_dict(orient="records")
        except Exception:
            pass

    for i, cfg in enumerate(grid, 1):
        if cfg.key() in done_keys:
            print("Skipping existing:", cfg)
            continue
        try:
            print(f"[{i}/{len(grid)}] {cfg}")
            res = run_single(cfg, args.doc, full_text, ref_text)
            results.append(res)
            pd.DataFrame(results).to_csv(args.out, index=False)
        except KeyboardInterrupt:
            print("Stopped by user. Saving partial results…")
            break
        except Exception as e:
            print(f"ERROR in run {i}: {e}")
            err_row = {**asdict(cfg), "error": str(e)}
            results.append(err_row)
            pd.DataFrame(results).to_csv(args.out, index=False)

    if results:
        df = pd.DataFrame(results)
        # Print best by each metric if present
        for metric in ["faithfulness", "rougeL", "bertscore", "answer_correctness"]:
            if metric in df.columns:
                best = df.sort_values(metric, ascending=False).head(1)
                print(f"\nBest by {metric}:\n", best.to_string(index=False))
        print(f"Saved {len(results)} rows to {args.out}")


if __name__ == "__main__":
    sys.exit(main())

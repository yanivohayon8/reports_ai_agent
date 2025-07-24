"""
Grid search for QnA RAG pipeline (RAGAS metrics).

Usage:
    python qna_grid_search.py \
        --doc data/report.pdf \
        --questions data/questions_answers.csv \
        --gt data/questions_answers.csv \
        --out results_qna.csv

What it does:
1. Builds the QnA tool for each hyper‑param combo (chunking + retrieval).
2. Runs all questions, collects answers & retrieved contexts.
3. Evaluates with RAGAS metrics (context_precision, context_recall, faithfulness by default).
4. Saves a CSV with the scores and params for each run (partial results are saved even on crash/CTRL+C).
"""
from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import sys
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Union

import pandas as pd
from datasets import Dataset
from ragas import evaluate as ragas_evaluate
import dotenv
from langchain_openai import ChatOpenAI  # Updated import

# ---- project imports (adjust paths to your repo) ----
from tools.tools_project.qna.qna_core import build_qna_tool  
from evaluators.core import RAGAS_METRICS_MAP, load_questions, read_any  

dotenv.load_dotenv()

DEFAULT_METRICS = ["context_precision", "context_recall", "faithfulness", "answer_correctness"]


# --------------------------------------------------------------------------- #
# Config dataclass for a single run                                           #
# --------------------------------------------------------------------------- #
@dataclass
class RunConfig:
    # chunking
    token_based: bool
    chunk_size: int
    chunk_overlap: int
    # retrieval
    top_k: int
    fetch_k: int
    lambda_mult: float
    score_threshold: Optional[float]
    use_multiquery: bool
    n_queries: int

    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)


# --------------------------------------------------------------------------- #
# Grid definition                                                             #
# --------------------------------------------------------------------------- #
def build_param_grid() -> list[RunConfig]:
    # ---- BEST BASE ----
    BASE = dict(
        token_based=True,
        chunk_size=1500,
        chunk_overlap=200,
        top_k=4,
        fetch_k=100,
        lambda_mult=0.5,
        score_threshold=None,
        use_multiquery=True,
        n_queries=8,
    )

    # ---- Want to ----
    sweep = {
        # param: list of candidate values
        "chunk_size":   [1500, 1800],
        "lambda_mult":  [0.3, 0.35, 0.5],
        "fetch_k":      [160, 200],
        "top_k":        [4],
        "n_queries":    [8, 10],
    }

    # ---  grid --- build the grid
    from itertools import product
    keys = list(sweep.keys())
    grid: list[RunConfig] = []
    for values in product(*[sweep[k] for k in keys]):
        cfg_args = BASE.copy()
        for k, v in zip(keys, values):
            cfg_args[k] = v
        grid.append(RunConfig(**cfg_args))
    return grid



# --------------------------------------------------------------------------- #
# Helpers                                                                      #
# --------------------------------------------------------------------------- #
def load_gt(gt_source: Union[str, pathlib.Path], questions: List[str]) -> List[str]:
    """Load per-question ground truth or duplicate a single text."""
    if isinstance(gt_source, pathlib.Path) and gt_source.exists():
        df = pd.read_csv(gt_source)
        if "question" not in df or "ground_truth" not in df:
            raise ValueError("GT CSV must have 'question' and 'ground_truth' columns")
        gt_map = dict(zip(df["question"], df["ground_truth"]))
        return [gt_map.get(q, "") for q in questions]
    return [str(gt_source)] * len(questions)


def build_ragas_dataset(
    questions: List[str],
    answers: List[str],
    contexts: List[List[str]],
    gts: List[str],
) -> Dataset:
    return Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": gts,
        }
    )


def normalize_scores(result, metrics: List[str]) -> Dict[str, float]:
    """
    RAGAS may return different structures (dict, EvaluationResult, list of Scores).
    Convert everything into plain floats per metric.
    """
    # Best path (new ragas) – pandas
    if hasattr(result, "to_pandas"):
        dfm = result.to_pandas()
        return {m: float(dfm[m].mean()) for m in metrics if m in dfm.columns}

    out: Dict[str, float] = {}
    for m in metrics:
        val = result[m]
        # Score object
        if hasattr(val, "value"):
            out[m] = float(val.value)
        # list of scores/values
        elif isinstance(val, list):
            vals = []
            for v in val:
                if hasattr(v, "value"):
                    vals.append(float(v.value))
                elif isinstance(v, (int, float)):
                    vals.append(float(v))
            out[m] = sum(vals) / len(vals) if vals else 0.0
        elif isinstance(val, (int, float)):
            out[m] = float(val)
        else:
            # last resort
            try:
                out[m] = float(val)
            except Exception:
                out[m] = 0.0
    return out


# --------------------------------------------------------------------------- #
# Single run                                                                   #
# --------------------------------------------------------------------------- #
def run_single(
    cfg: RunConfig,
    full_text: str,
    questions: List[str],
    gts: List[str],
    metrics: List[str],
    model_name: str,
    temperature: float,
) -> Dict[str, Any]:
    """Run one configuration and return scores + params."""

    tool = build_qna_tool(
        [full_text],
        token_based=cfg.token_based,
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        top_k=cfg.top_k,
        fetch_k=cfg.fetch_k,
        lambda_mult=cfg.lambda_mult,
        score_threshold=cfg.score_threshold,
        use_multiquery=cfg.use_multiquery,
        n_queries=cfg.n_queries,
        llm_model=model_name,
        temperature=temperature,
        return_sources=False,
    )

    retriever = tool.retriever

    answers: List[str] = []
    contexts: List[List[str]] = []
    for q in questions:
        res = tool.run(q)
        answers.append(res["answer"])
        # LangChain >=0.2: use invoke instead of get_relevant_documents
        try:
            raw_docs = retriever.invoke(q)
        except AttributeError:
            raw_docs = retriever.get_relevant_documents(q)
        contexts.append([d.page_content for d in raw_docs])

    dataset = build_ragas_dataset(questions, answers, contexts, gts)
    metric_fns = [RAGAS_METRICS_MAP[m] for m in metrics]

    ragas_res = ragas_evaluate(dataset, metrics=metric_fns, raise_exceptions=False)

    scores = normalize_scores(ragas_res, metrics)
    out = {**scores, **asdict(cfg)}
    return out


# --------------------------------------------------------------------------- #
# Main                                                                         #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Grid search QnA hyper-params")
    ap.add_argument("--doc", type=pathlib.Path, required=True, help="Knowledge source (txt/pdf)")
    ap.add_argument("--questions", type=pathlib.Path, required=True, help="CSV of questions")
    ap.add_argument("--gt", type=pathlib.Path, help="CSV with ground_truth per question, or omit to use doc")
    ap.add_argument("--out", type=pathlib.Path, default=pathlib.Path("grid_results.csv"))
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--metrics", default="context_precision,context_recall,faithfulness, answer_correctness")
    ap.add_argument("--max-runs", type=int, default=None, help="Limit number of configs (debug)")
    args = ap.parse_args()

    full_text = read_any(args.doc)
    questions = load_questions(args.questions)
    gt_source: Union[str, pathlib.Path] = args.gt if args.gt else full_text
    gts = load_gt(gt_source, questions)

    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    grid = build_param_grid()
    if args.max_runs:
        grid = grid[: args.max_runs]

    results: List[Dict[str, Any]] = []
    for i, cfg in enumerate(grid, 1):
        try:
            print(f"[{i}/{len(grid)}] running {cfg}")
            res = run_single(cfg, full_text, questions, gts, metrics, args.model, args.temperature)
            results.append(res)
            pd.DataFrame(results).to_csv(args.out, index=False)
        except KeyboardInterrupt:
            print("Stopped by user. Saving partial results...")
            break
        except Exception as e:  # noqa: BLE001
            print(f"ERROR in run {i}: {e}")
            err_row = {**asdict(cfg), "error": str(e)}
            results.append(err_row)
            pd.DataFrame(results).to_csv(args.out, index=False)

    if results:
        df = pd.DataFrame(results)
        if "context_recall" in df.columns:
            best = df.sort_values("context_recall", ascending=False).head(1)
            print("\nBest by recall:\n", best.to_string(index=False))
        print(f"Saved {len(results)} rows to {args.out}")


if __name__ == "__main__":
    sys.exit(main())

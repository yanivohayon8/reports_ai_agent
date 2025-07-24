#!/usr/bin/env python
"""Unified CLI wrapper for QnA & Summary evaluators."""
from __future__ import annotations

import argparse
import pathlib

from evaluators.core import read_any
from evaluators.qna_eval import QnaEvaluator
from evaluators.summary_eval import SummaryEvaluator


def parse_args():
    p = argparse.ArgumentParser(description="Batch evaluator for QnA & Summary")
    p.add_argument("doc", type=pathlib.Path, help="Source knowledge doc (txt/pdf)")

    p.add_argument("--doc-gt", type=pathlib.Path,
                   help="Ground-truth file (txt/pdf) used for ALL questions (fallback if --gt not given)")
    p.add_argument("--gt", type=pathlib.Path,
                   help="CSV with per-question ground_truth (columns: question,ground_truth)")

    p.add_argument("--questions", "-q", type=pathlib.Path, help="Questions file")
    p.add_argument("--answers-out", "-ao", type=pathlib.Path, default=pathlib.Path("answers.csv"))
    p.add_argument("--qa-metrics", default="context_precision,context_recall,faithfulness")
    p.add_argument("--top-k", type=int, default=12)
    p.add_argument("--qna-only", action="store_true")

    p.add_argument("--summary-ref", type=pathlib.Path, help="Reference summary for ROUGE/BERT")
    p.add_argument("--summary-out", "-so", type=pathlib.Path, default=pathlib.Path("summary.txt"))
    p.add_argument("--summary-metrics", default="faithfulness,rougeL")
    p.add_argument("--summary-only", action="store_true")

    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.0)
    return p.parse_args()


def main():
    args = parse_args()

    run_qna = (not args.summary_only) and args.questions is not None
    run_sum = (not args.qna_only) and args.summary_ref is not None
    if not run_qna and not run_sum:
        raise SystemExit("Nothing to do – supply questions and/or choose pipeline")

    # Fallback GT text (single big text) if no per-question CSV
    fallback_gt = read_any(args.doc_gt) if args.doc_gt else read_any(args.doc)

    if run_qna:
        qna_eval = QnaEvaluator(args.doc, model_name=args.model, temperature=args.temperature, top_k=args.top_k)

        # Decide what to pass as gt_source
        gt_source = args.gt if args.gt else fallback_gt

        qa_scores = qna_eval.run_batch(
            questions_file=args.questions,
            answers_out=args.answers_out,
            gt_source=gt_source,
            # metrics=[m.strip() for m in args.qa_metrics.split(",") if m.strip()],
        )
        print("QnA scores →", qa_scores)

    if run_sum:
        sum_eval = SummaryEvaluator(args.doc, model_name=args.model, temperature=args.temperature)
        summary_txt = sum_eval.generate()
        args.summary_out.write_text(summary_txt, encoding="utf-8")
        print(f"Summary saved to {args.summary_out}")

        sum_scores = sum_eval.evaluate(
            summary=summary_txt,
            ground_text=fallback_gt,
            ref_summary_path=args.summary_ref,
            metrics=[m.strip() for m in args.summary_metrics.split(",") if m.strip()],
        )
        print("Summary scores →", sum_scores)


if __name__ == "__main__":
    main()

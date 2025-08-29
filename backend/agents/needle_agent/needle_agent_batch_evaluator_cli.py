import sys
import argparse
from pathlib import Path

sys.path.append(".")

from needle_agent_batch_evaluator import (
    evaluate_dataset,
    combine_datasets,
    init_needle_agent,
)
from backend.core.config_utils import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Custom Needle Agent Batch Evaluator CLI (Final Results Only)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python needle_agent_batch_evaluator_custom_cli.py
  python needle_agent_batch_evaluator_custom_cli.py --skip-combined
        """,
    )

    parser.add_argument(
        "--config-path",
        type=str,
        default="backend/agents/needle_agent/needle_evaluator_config.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--evaluation-dir",
        type=str,
        default="C:/Users/HP/Documents/reports_ai_agent/backend/data/evaluation_datasets/needle_agent",
        help="Directory containing .jsonl datasets",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="C:/Users/HP/Documents/reports_ai_agent/backend/data/evaluation_datasets/needle_agent/results",
        help="Directory where evaluation results will be saved",
    )
    parser.add_argument(
        "--skip-combined",
        action="store_true",
        help="Skip creating and evaluating the combined dataset",
    )

    args = parser.parse_args()

    evaluation_dir = Path(args.evaluation_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load config and init agent
    config = load_config(args.config_path)
    needle_agent = init_needle_agent(config)

    # Step 1: evaluate each dataset individually
    dataset_paths = sorted(evaluation_dir.glob("*.jsonl"))
    if not dataset_paths:
        print(f"✗ No datasets found in {evaluation_dir}")
        sys.exit(1)

    successful = []
    failed = []

    for dataset_path in dataset_paths:
        ok = evaluate_dataset(config, str(dataset_path), results_dir, needle_agent)
        if ok:
            successful.append(dataset_path.name)
        else:
            failed.append(dataset_path.name)

    # Step 2: combine and evaluate once if not skipped
    if not args.skip_combined:
        combined_path = results_dir / "combined_dataset.jsonl"
        combine_datasets(dataset_paths, combined_path)
        ok = evaluate_dataset(config, str(combined_path), results_dir, needle_agent)
        if ok:
            successful.append("combined_dataset.jsonl")
        else:
            failed.append("combined_dataset.jsonl")

    # --- FINAL SUMMARY ---
    print("\n===============================")
    print("EVALUATION COMPLETE - SUMMARY")
    print("===============================")
    print(f" Total datasets processed: {len(dataset_paths)}")
    if not args.skip_combined:
        print(f" +1 combined dataset included")
    print(f" Successful evaluations: {len(successful)}")
    print(f" Failed evaluations:     {len(failed)}")
    print(f" Results saved in:       {results_dir}")
    if failed:
        print("\nFailed datasets:")
        for name in failed:
            print(f"  - {name}")
    print("===============================\n")


if __name__ == "__main__":
    main()

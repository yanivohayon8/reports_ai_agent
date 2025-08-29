import argparse
from pathlib import Path
import asyncio
import sys

sys.path.append(".")

from backend.agents.router_agent.router_dataset_evaluator import RouterDatasetEvaluator
from backend.agents.router_agent.router import RouterAgent


def main():
    parser = argparse.ArgumentParser(description="Router dataset evaluator CLI")
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, help="Where to save the evaluation JSON")
    args = parser.parse_args()

    evaluator = RouterDatasetEvaluator(Path(args.dataset_path))
    router = RouterAgent()

    result = asyncio.run(evaluator.evaluate(router))

    out = Path(args.output_path) if args.output_path else Path(args.dataset_path).with_suffix("")
    if out.is_dir():
        out_file = out / (Path(args.dataset_path).stem + "_router_results.json")
    else:
        out_file = Path(str(out) + "_router_results.json")

    evaluator.save_results(result, out_file)
    print(f"✓ Saved evaluation results to {out_file}")


if __name__ == "__main__":
    main()



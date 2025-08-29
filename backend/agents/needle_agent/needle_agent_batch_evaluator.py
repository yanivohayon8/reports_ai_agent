import sys
import shutil
from pathlib import Path

sys.path.append(".")

from backend.agents.needle_agent.needle_evaluator import NeedleEvaluator
from backend.agents.needle_agent.needle_agent import NeedleAgent
from backend.core.api_utils import get_llm_langchain_openai
from backend.indexer.indexer import FAISSIndexer


def init_needle_agent(config: dict) -> NeedleAgent:
    llm = get_llm_langchain_openai(model=config["NeedleAgent"]["llm"]["model"])
    indexer = FAISSIndexer.from_small_embedding(
        directory_path=config["NeedleAgent"]["indexer"]["faiss_directory"]
    )
    return NeedleAgent(indexer, llm)


def init_needle_evaluator(config: dict, dataset_path: str) -> NeedleEvaluator:
    llm = get_llm_langchain_openai(model=config["Evaluator"]["evaluator_llm"])
    metrics = config["Evaluator"]["metrics"]
    return NeedleEvaluator(Path(dataset_path), llm, selected_metrics=metrics)


def evaluate_dataset(config: dict, dataset_path: str, output_dir: Path, needle_agent: NeedleAgent) -> bool:
    """Evaluate a single dataset and save results to output_dir"""
    dataset_name = Path(dataset_path).stem
    try:
        evaluator = init_needle_evaluator(config, dataset_path)
        result = evaluator.evaluate(needle_agent)
        output_file = output_dir / f"{dataset_name}_results.json"
        evaluator.save_results(result, output_file, agent=needle_agent)
        return True
    except Exception as e:
        print(f"✗ Error evaluating {dataset_name}: {e}")
        return False


def combine_datasets(dataset_paths, combined_path: Path):
    """Combine multiple JSONL datasets into one"""
    with open(combined_path, "w", encoding="utf-8") as outfile:
        for path in dataset_paths:
            with open(path, "r", encoding="utf-8") as infile:
                shutil.copyfileobj(infile, outfile)
    return combined_path

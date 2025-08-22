import sys
sys.path.append(".")

from agents.needle_agent.needle_evaluator import NeedleEvaluator
from pathlib import Path
from core.api_utils import get_llm_langchain_openai
from agents.needle_agent.needle_agent import NeedleAgent
from core.config_utils import load_config
import argparse
from indexer.indexer import FAISSIndexer


def init_needle_agent(config: dict):
    llm = get_llm_langchain_openai(model=config["NeedleAgent"]["llm"]["model"])
    indexer = FAISSIndexer.from_small_embedding(directory_path=config["NeedleAgent"]["indexer"]["faiss_directory"])
    return NeedleAgent(indexer, llm)

def init_needle_evaluator(config: dict):
    ground_truth_dataset_path = Path(config["Dataset"]["ground_truth_dataset_path"])
    llm = get_llm_langchain_openai(model=config["Evaluator"]["evaluator_llm"])
    metrics = config["Evaluator"]["metrics"]

    return NeedleEvaluator(ground_truth_dataset_path, llm, selected_metrics=metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="agents/needle_agent/needle_evaluator_config.yaml")
    args = parser.parse_args()

    config = load_config(args.config_path)

    needle_agent = init_needle_agent(config)
    needle_evaluator = init_needle_evaluator(config)
    result = needle_evaluator.evaluate(needle_agent)

    print("-"*50 + "result" + "-"*50)
    print(result)


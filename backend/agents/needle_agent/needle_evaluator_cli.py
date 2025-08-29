import sys
sys.path.append(".")

from backend.agents.needle_agent.needle_evaluator import NeedleEvaluator
from pathlib import Path
from backend.core.api_utils import get_llm_langchain_openai
from backend.agents.needle_agent.needle_agent import NeedleAgent
from backend.core.config_utils import load_config
import argparse
from backend.indexer.indexer import FAISSIndexer
import json


def init_needle_agent(config: dict):
    llm = get_llm_langchain_openai(model=config["NeedleAgent"]["llm"]["model"])
    indexer = FAISSIndexer.from_small_embedding(directory_path=config["NeedleAgent"]["indexer"]["faiss_directory"])
    return NeedleAgent(indexer, llm)

def init_needle_evaluator(config: dict):
    ground_truth_dataset_path = Path(config["Dataset"]["ground_truth_dataset_path"])
    llm = get_llm_langchain_openai(model=config["Evaluator"]["evaluator_llm"])
    metrics = config["Evaluator"]["metrics"]

    return NeedleEvaluator(ground_truth_dataset_path, llm, selected_metrics=metrics)

def evaluate_single_dataset(config: dict, dataset_path: str):
    """Evaluate a single dataset"""
    # Temporarily update the config for this dataset
    temp_config = config.copy()
    temp_config["Dataset"]["ground_truth_dataset_path"] = dataset_path
    
    needle_agent = init_needle_agent(temp_config)
    needle_evaluator = init_needle_evaluator(temp_config)
    result = needle_evaluator.evaluate(needle_agent)
    
    return result, needle_evaluator

def evaluate_all_in_directory(config: dict, directory_path: str):
    """Evaluate all .jsonl files in a directory"""
    directory = Path(directory_path)
    datasets = list(directory.glob("*.jsonl"))
    
    if not datasets:
        print(f"No .jsonl files found in {directory_path}")
        return {}
    
    print(f"Found {len(datasets)} datasets to evaluate:")
    for dataset in datasets:
        print(f"  - {dataset.name}")
    
    all_results = {}
    
    for dataset_path in datasets:
        print(f"\n{'='*60}")
        print(f"Evaluating: {dataset_path.name}")
        print(f"{'='*60}")
        
        try:
            result, evaluator = evaluate_single_dataset(config, str(dataset_path))
            
            # Store results
            dataset_name = dataset_path.stem
            all_results[dataset_name] = result
            
            # Save individual results
            output_path = directory / f"{dataset_name}_evaluation_results.json"
            evaluator.save_results(result, output_path, agent=None)
            
            print(f"Results for {dataset_name}:")
            print(result)
            print(f"Results saved to: {output_path}")
            
        except Exception as e:
            print(f"Error evaluating {dataset_path}: {e}")
            all_results[dataset_path.stem] = {"error": str(e)}
    
    # Save combined results
    combined_output_path = directory / "all_datasets_evaluation_results.json"
    with open(combined_output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("EVALUATION COMPLETE")
    print(f"{'='*60}")
    print(f"Combined results saved to: {combined_output_path}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", type=str, default="backend/agents/needle_agent/needle_evaluator_config.yaml")
    parser.add_argument("--evaluate-all", action="store_true", help="Evaluate all .jsonl files in the evaluation directory")
    parser.add_argument("--evaluation-dir", type=str, default="backend/data/evaluation_datasets/needle_agent", 
                       help="Directory containing evaluation datasets")
    args = parser.parse_args()

    config = load_config(args.config_path)

    if args.evaluate_all:
        # Evaluate all datasets in the directory
        evaluate_all_in_directory(config, args.evaluation_dir)
    else:
        # Evaluate single dataset (original behavior)
        needle_agent = init_needle_agent(config)
        needle_evaluator = init_needle_evaluator(config)
        result = needle_evaluator.evaluate(needle_agent)

        print("-"*50 + "result" + "-"*50)
        print(result)
        
        needle_evaluator.save_results(result, Path(config["Evaluator"]["output_path"]), 
                                      agent=needle_agent)



from agents.needle_agent.needle_evaluator import NeedleEvaluator
from agents.needle_agent.needle_agent import NeedleAgent
from pathlib import Path
from core.api_utils import get_llm_langchain_openai
from indexer.indexer import FAISSIndexer
import json
import shutil
from ragas.evaluation import EvaluationResult,EvaluationDataset
import pandas as pd

def test_supported_metrics():
    ground_truth_dataset_path = Path("tests/data/evaluation_datasets/needle_agent/client2_report2_tourAndCarePolicy.jsonl")
    metrics = ["Faithfulness","LLMContextRecall","LLMContextPrecisionWithReference"]
    llm = get_llm_langchain_openai(model="gpt-4o-mini")
    needle_evaluator = NeedleEvaluator(ground_truth_dataset_path,llm,selected_metrics=metrics)

    # TODO: create a naive indexer for testing
    faiss_indexer_path = Path("vectordb_indexes/faiss_indexer_insurance")
    faiss_indexer = FAISSIndexer.from_small_embedding(directory_path=faiss_indexer_path)

    needle_agent = NeedleAgent(faiss_indexer, llm)
    result = needle_evaluator.evaluate(needle_agent)

    assert result is not None
    print(result)

def test_none_metrics():
    ground_truth_dataset_path = Path("tests/data/evaluation_datasets/needle_agent/client2_report2_tourAndCarePolicy.jsonl")
    llm = get_llm_langchain_openai(model="gpt-4o-mini")
    needle_evaluator = NeedleEvaluator(ground_truth_dataset_path, llm,selected_metrics=None)

    # TODO: create a naive indexer for testing
    faiss_indexer_path = Path("vectordb_indexes/faiss_indexer_insurance")
    faiss_indexer = FAISSIndexer.from_small_embedding(directory_path=faiss_indexer_path)

    needle_agent = NeedleAgent(faiss_indexer, llm)
    result = needle_evaluator.evaluate(needle_agent)

    assert result is not None
    print(result)

    output_path = Path("tests/data/evaluation_datasets/needle_agent/client2_report2_tourAndCarePolicy_evaluation_results.json")
    needle_evaluator.save_results(result, output_path, agent=needle_agent)

    assert output_path.exists()
    assert output_path.is_file()

    with open(output_path, "r") as f:
        saved_results = json.load(f)

    assert saved_results is not None

    shutil.rmtree(output_path)


from agents.needle_agent.needle_evaluator import NeedleEvaluator
from agents.needle_agent.needle import NeedleAgent
from pathlib import Path
from core.api_utils import get_llm_langchain_openai
from indexer.indexer import FAISSIndexer

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
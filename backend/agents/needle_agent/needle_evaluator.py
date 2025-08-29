from backend.agents.needle_agent.needle_dataset_synthesizer import load_needle_dataset
from pathlib import Path
from backend.agents.needle_agent.needle_agent import NeedleAgent
from typing import List
from ragas.metrics import (
    Faithfulness,
    LLMContextRecall,
    LLMContextPrecisionWithReference
)

from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas import EvaluationDataset,evaluate as ragas_evaluate
from ragas.evaluation import EvaluationResult
from pandas import DataFrame
import json

SUPPORTED_METRICS = [Faithfulness,LLMContextRecall,LLMContextPrecisionWithReference]

class NeedleEvaluator:
    def __init__(self, ground_truth_dataset_path: Path, llm:ChatOpenAI, selected_metrics:List[str]=None):
        self.ground_truth_dataset = load_needle_dataset(ground_truth_dataset_path)        
        self.metrics = None # let ragas decide (not recommended)

        if selected_metrics:
            self.metrics = [supported_metric() for supported_metric in SUPPORTED_METRICS \
                            if supported_metric.__name__ in selected_metrics]

        self.evaluator_llm = LangchainLLMWrapper(llm)

    def evaluate(self, agent: NeedleAgent):
        dataset = self._generate_dataset(agent)
        result = ragas_evaluate(dataset,metrics=self.metrics,llm=self.evaluator_llm)  

        return result


    def _generate_dataset(self, agent: NeedleAgent):
        dataset = []

        for ground_truth_sample in self.ground_truth_dataset:
            agent_response = agent.answer(ground_truth_sample["question"])
            sample = {
                "user_input": ground_truth_sample["question"],
                "retrieved_contexts": agent_response.get("chunks_content", []),
                "response": agent_response["answer"],
                "reference": ground_truth_sample["answer"]
            }

            dataset.append(sample)
        
        return EvaluationDataset.from_list(dataset)
            
    def save_results(self, evaluation_result: EvaluationResult, output_path: Path, agent: NeedleAgent=None):
        df_results = evaluation_result.to_pandas()
        mean_scores = self._compute_mean(df_results)

        if agent:
            agent_input = agent.get_used_input()
        else:
            agent_input = None

        saved_results = {
            "results": {
                "mean_scores": mean_scores,
            },
            "agent_input": agent_input
        }

        with open(output_path, "w") as f:
            json.dump(saved_results, f)

    def _compute_mean(self,df:DataFrame)->dict:
        try:
            df = df.drop(columns=["user_input"])
        except KeyError:
            pass
        try:
            df = df.drop(columns=["retrieved_contexts"])
        except KeyError:
            pass
        try:
            df = df.drop(columns=["response"])
        except KeyError:
            pass
        try:
            df = df.drop(columns=["reference"])
        except KeyError:
            pass

        means = df.mean(axis=0)

        return means.to_dict()



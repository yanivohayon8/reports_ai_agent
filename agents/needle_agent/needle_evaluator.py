from agents.needle_agent.needle_dataset_synthesizer import load_needle_dataset
from pathlib import Path
from agents.needle_agent.needle import NeedleAgent
from typing import List
from ragas.metrics import (
    Faithfulness,
    LLMContextRecall,
    LLMContextPrecisionWithReference
)

from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas import EvaluationDataset,evaluate as ragas_evaluate


SUPPORTED_METRICS = [Faithfulness,LLMContextRecall,LLMContextPrecisionWithReference]

class NeedleEvaluator:
    def __init__(self, ground_truth_dataset_path: Path, llm:ChatOpenAI, selected_metrics:List[str]=None):
        self.ground_truth_dataset = load_needle_dataset(ground_truth_dataset_path)        
        self.metrics = None # let ragas decide (not recommended)

        if selected_metrics:
            metrics_titles = [metric.title() for metric in selected_metrics]
            self.metrics = [supported_metric() for supported_metric in SUPPORTED_METRICS if supported_metric.__name__ in metrics_titles]

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
                "retrieved_contexts": agent_response["chunks_content"],
                "response": agent_response["answer"],
                "reference": ground_truth_sample["answer"]
            }

            dataset.append(sample)
        
        return EvaluationDataset.from_list(dataset)


            




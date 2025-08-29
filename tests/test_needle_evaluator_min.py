import json
from pathlib import Path
from types import SimpleNamespace, ModuleType
import sys

import pandas as pd
import pytest

# --- Provide lightweight stand-ins for ragas to satisfy imports ---
ragas = ModuleType("ragas")
ragas_metrics = ModuleType("ragas.metrics")
ragas_llms = ModuleType("ragas.llms")
ragas_evaluation = ModuleType("ragas.evaluation")

class _Metric:  # placeholder metric classes
    __name__ = "Metric"

class Faithfulness(_Metric):
    __name__ = "Faithfulness"

class LLMContextRecall(_Metric):
    __name__ = "LLMContextRecall"

class LLMContextPrecisionWithReference(_Metric):
    __name__ = "LLMContextPrecisionWithReference"

class LangchainLLMWrapper:  # placeholder wrapper
    def __init__(self, llm):
        self.llm = llm

class EvaluationDataset:
    @staticmethod
    def from_list(items):
        return SimpleNamespace(items=items)

class EvaluationResult:
    def __init__(self, df: pd.DataFrame):
        self._df = df
    def to_pandas(self):
        return self._df

# Attach to modules
ragas_metrics.Faithfulness = Faithfulness
ragas_metrics.LLMContextRecall = LLMContextRecall
ragas_metrics.LLMContextPrecisionWithReference = LLMContextPrecisionWithReference
ragas_llms.LangchainLLMWrapper = LangchainLLMWrapper
ragas_evaluation.EvaluationDataset = EvaluationDataset
ragas_evaluation.EvaluationResult = EvaluationResult

# Provide top-level names as imported by production code
ragas.EvaluationDataset = EvaluationDataset

def _fake_evaluate(dataset, metrics=None, llm=None):
    return SimpleNamespace(result="ok")
ragas.evaluate = _fake_evaluate

# Register in sys.modules prior to importing target
sys.modules.setdefault("ragas", ragas)
sys.modules.setdefault("ragas.metrics", ragas_metrics)
sys.modules.setdefault("ragas.llms", ragas_llms)
sys.modules.setdefault("ragas.evaluation", ragas_evaluation)

# ------------------------------------------------------------------
# Import target after stubbing ragas
import backend.agents.needle_agent.needle_evaluator as ne


class DummyAgent:
    def __init__(self, answer_text: str = "ans", chunks=None):
        self._answer_text = answer_text
        self._chunks = chunks or ["c1", "c2"]

    def answer(self, q: str):
        return {
            "answer": self._answer_text,
            "chunks_content": list(self._chunks),
            "chunks_metadata": [{"page": 1}, {"page": 2}],
        }

    def get_used_input(self):
        return {"faiss_indexer_input": {"index": "needle"}, "llm_model": {"model": "m"}}


@pytest.fixture
def tmp_json(tmp_path: Path):
    return tmp_path / "out.json"


def test_generate_dataset_and_evaluate(monkeypatch):
    ground_truth = [
        {"question": "q1", "answer": "a1"},
        {"question": "q2", "answer": "a2"},
    ]

    monkeypatch.setattr(ne, "load_needle_dataset", lambda p: ground_truth)

    captured_samples = {}

    class DummyEvalDataset:
        pass

    def fake_from_list(samples):
        captured_samples["samples"] = samples
        return DummyEvalDataset()

    monkeypatch.setattr(ne.EvaluationDataset, "from_list", staticmethod(fake_from_list))

    called = {}

    def fake_ragas_evaluate(dataset, metrics=None, llm=None):
        called["dataset"] = dataset
        called["metrics"] = metrics
        called["llm"] = llm
        return SimpleNamespace(result="ok")

    monkeypatch.setattr(ne, "ragas_evaluate", fake_ragas_evaluate)

    evaluator = ne.NeedleEvaluator(Path("dummy.jsonl"), llm=SimpleNamespace(), selected_metrics=["Faithfulness"]) 

    agent = DummyAgent()

    result = evaluator.evaluate(agent)

    assert isinstance(captured_samples["samples"], list)
    assert captured_samples["samples"][0]["user_input"] == "q1"
    assert captured_samples["samples"][0]["retrieved_contexts"] == ["c1", "c2"]
    assert captured_samples["samples"][0]["response"] == "ans"
    assert captured_samples["samples"][0]["reference"] == "a1"

    assert hasattr(result, "result") and result.result == "ok"


def test_save_results_writes_mean_scores_and_agent_input(monkeypatch, tmp_json):
    evaluator = ne.NeedleEvaluator.__new__(ne.NeedleEvaluator)

    df = pd.DataFrame([
        {"user_input": "q", "retrieved_contexts": ["c"], "response": "r", "reference": "ref",
         "faithfulness": 0.5, "context_recall": 0.25}
    ])

    class DummyEvalResult:
        def to_pandas(self):
            return df.copy()

    dummy_agent = DummyAgent()

    evaluator.save_results(DummyEvalResult(), tmp_json, agent=dummy_agent)

    data = json.loads(tmp_json.read_text())
    assert "results" in data and "mean_scores" in data["results"]
    assert pytest.approx(data["results"]["mean_scores"]["faithfulness"], rel=1e-6) == 0.5
    assert data["agent_input"]["llm_model"]["model"] == "m"


def test_compute_mean_drops_non_metric_columns():
    evaluator = ne.NeedleEvaluator.__new__(ne.NeedleEvaluator)
    df = pd.DataFrame([
        {"user_input": "u", "retrieved_contexts": ["c"], "response": "r", "reference": "ref",
         "metric1": 0.1, "metric2": 0.3}
    ])
    means = evaluator._compute_mean(df)
    assert set(means.keys()) == {"metric1", "metric2"}
    assert means["metric1"] == 0.1 and means["metric2"] == 0.3

from pathlib import Path
from typing import List
import json

from backend.agents.router_agent.router_dataset_synthesizer import load_router_dataset
from backend.agents.router_agent.router import RouterAgent


class RouterDatasetEvaluator:
    def __init__(self, dataset_path: Path):
        self.rows = load_router_dataset(dataset_path)

    async def evaluate(self, router: RouterAgent) -> dict:
        total = 0
        correct = 0
        details: List[dict] = []
        for row in self.rows:
            q = row.get("question", "")
            expected = row.get("expected_type")
            total += 1
            try:
                cls = await router.classify(q)
                got = cls.type
                ok = str(got) == str(expected)
            except Exception:
                got = "error"
                ok = False
            correct += 1 if ok else 0
            details.append({"question": q, "expected": expected, "predicted": got, "ok": ok})
        accuracy = correct / total if total else 0.0
        return {"accuracy": accuracy, "total": total, "correct": correct, "details": details}

    def save_results(self, result: dict, output_path: Path):
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)



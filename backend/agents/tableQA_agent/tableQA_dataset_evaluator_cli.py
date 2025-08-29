import argparse
from pathlib import Path
import sys
sys.path.append(".")

from backend.core.config_utils import load_config
from backend.core.api_utils import get_llm_langchain_openai
from backend.agents.tableQA_agent.tableQA_agent import TableQAAgent
from backend.agents.tableQA_agent.tableQA_dataset_evaluator import TableQADatasetEvaluator
from backend.indexer.indexer import FAISSIndexer
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main():
    parser = argparse.ArgumentParser(description="TableQA dataset evaluator CLI")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to .jsonl dataset")
    parser.add_argument("--config-path", type=str, default="backend/agents/tableQA_agent/tableQA_dataset_evaluator_config.yaml")
    parser.add_argument("--output-path", type=str, help="Where to save results JSON")
    args = parser.parse_args()

    cfg = load_config(args.config_path)

    # Init agent components
    faiss_dir = Path(cfg["TableQAAgent"]["indexer"]["faiss_directory"]).resolve()
    faiss = FAISSIndexer.from_small_embedding(directory_path=str(faiss_dir))

    chunk_cfg = cfg["TableQAAgent"]["chunker"]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_cfg.get("chunk_size", 1200),
        chunk_overlap=chunk_cfg.get("chunk_overlap", 200),
        separators=chunk_cfg.get("separators", ["\n\n", "\n", " ", ""]) 
    )

    llm = get_llm_langchain_openai(model=cfg["TableQAAgent"]["llm"]["model"])
    agent = TableQAAgent(faiss_indexer=faiss, text_splitter=splitter, llm=llm)

    evaluator_llm = get_llm_langchain_openai(model=cfg["evaluator"]["llm"]["model"])
    evaluator = TableQADatasetEvaluator(Path(args.dataset_path), llm=evaluator_llm, selected_metrics=cfg.get("metrics"))

    result = evaluator.evaluate(agent)

    output_path = Path(args.output_path) if args.output_path else Path(args.dataset_path).with_suffix("")
    if output_path.is_dir():
        out_file = output_path / (Path(args.dataset_path).stem + "_results.json")
    else:
        out_file = Path(str(output_path) + "_results.json")

    evaluator.save_results(result, out_file, agent=agent)
    print(f"✓ Saved evaluation results to {out_file}")


if __name__ == "__main__":
    main()

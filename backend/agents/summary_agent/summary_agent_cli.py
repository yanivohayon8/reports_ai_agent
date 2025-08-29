import argparse
import yaml
from pathlib import Path
import sys
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Ensure project root is on sys.path when running via Debug/Run (script mode)
# Ensure project root is on path (so 'backend' package is importable)
FILE_PATH = Path(__file__).resolve()
PROJECT_ROOT = FILE_PATH.parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.agents.summary_agent.summary_chunker import SummaryChunker
from backend.agents.summary_agent.summary_agent import SummaryAgent
from backend.indexer.indexer import FAISSIndexer
from backend.core.api_utils import get_llm_langchain_openai
from backend.core.user_interface import ConsoleChat


def load_config(config_path: str):
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return yaml.safe_load(config_file.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(
        description="CLI for SummaryAgent",
        epilog="Example: python summary_agent_cli.py --pdf tests/data/sample.pdf --method map_reduce"
    )
    parser.add_argument("--pdf", type=str, required=True, help="Path to PDF file to summarize")
    parser.add_argument("--method", type=str, choices=["map_reduce", "iterative", "query_based"], default="map_reduce")
    parser.add_argument("--query", type=str, help="Query for query_based summarization")
    parser.add_argument("--config", type=str, default="backend/agents/summary_agent/config.yaml")

    args = parser.parse_args()
    config = load_config(args.config)

    # --- Init FAISS indexer ---
    indexer_conf = config["SummaryAgent"]["indexer"]
    faiss_indexer = FAISSIndexer.from_small_embedding(directory_path=indexer_conf["faiss_directory"])

    # --- Init splitter ---
    chunker_conf = config["SummaryAgent"]["chunker"]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunker_conf.get("chunk_size", 1200),
        chunk_overlap=chunker_conf.get("chunk_overlap", 200),
        separators=chunker_conf.get("separators", ["\n\n", "\n", " ", ""])
    )

    # --- Init Chunker ---
    chunker = SummaryChunker(faiss_indexer=faiss_indexer, text_splitter=splitter)

    # --- Init LLM ---
    llm_conf = config["SummaryAgent"]["llm"]
    llm = get_llm_langchain_openai(model=llm_conf["model"])

    # --- Init Agent ---
    agent = SummaryAgent(chunker=chunker, llm=llm)

    # --- Run summarization ---
    pdf_path = Path(args.pdf)
    result = agent.summarize_single_pdf(pdf_path, method=args.method, query=args.query)

    # --- Print result ---
    print(f"\n=== Summary using {args.method} ===\n")
    print(result.get("answer", "No answer generated"))

    if "chunks_summary" in result:
        print("\n--- Partial summaries ---")
        for i, chunk_sum in enumerate(result["chunks_summary"], 1):
            print(f"[Chunk {i}] {chunk_sum}")

    # --- Re-enable interactive console chat ---
    try:
        # Build context from chunks for interactive query-based summaries
        docs = chunker.chunk(pdf_path)
        context_text = "\n\n".join(d.page_content for d in docs)

        def processor_func(user_message: str) -> str:
            out = agent.summarize_with_query(user_message, context_text)
            return out.get("answer", "")

        chat = ConsoleChat(processor_func)
        chat.start()
    except Exception as e:
        print(f"\n[ConsoleChat disabled due to error: {e}]")

if __name__ == "__main__":
    main()

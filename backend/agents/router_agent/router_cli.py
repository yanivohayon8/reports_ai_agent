import sys
import asyncio
from pathlib import Path
from langchain_core.documents import Document

sys.path.append("./")

import sys
import os

# Add the backend directory to Python path
backend_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, backend_path)

from core.user_interface import ConsoleChat
from agents.router_agent.router import RouterAgent
from retrieval import DenseRetriever, SparseRetriever, HybridRetriever
from indexer.indexer import FAISSIndexer
from core.config_utils import load_config
from core.api_utils import get_llm_langchain_openai
from core.pdf_reader import read_pdf  # assuming you already have this utility


def load_docs_from_path(path: str):
    """Load all PDF docs from the given directory. Fallback to a tiny dummy table."""
    try:
        docs = []
        base = Path(path)
        if base.exists():
            for pdf_file in base.glob("*.pdf"):
                try:
                    pdf_docs = read_pdf(pdf_file, format="documents")
                    docs.extend(pdf_docs)
                except Exception:
                    # If parsing fails, add filename as a stub document
                    docs.append(Document(page_content=f"Parsed placeholder for {pdf_file.name}"))
        if docs:
            return docs
    except Exception:
        pass
    # Fallback: minimal table doc for TableQA
    return [Document(page_content="| Name | Salary |\n| Alice | 5000 |\n| Bob | 7000 |")]


def build_router():
    # Load config with safe defaults
    try:
        config = load_config("agents/router_agent/config.yaml")
    except Exception:
        config = {"faiss_indexer": {"directory": "vector_db"}, "llm": {"model": "gpt-4o-mini"}}

    # Build FAISS index for dense retrieval
    faiss_config = config.get("faiss_indexer", {})
    faiss_dir = faiss_config.get("directory", "vector_db")
    faiss_indexer = FAISSIndexer.from_small_embedding(directory_path=faiss_dir)

    dense = DenseRetriever(faiss_indexer.vector_store)

    # Load docs for sparse retriever (configurable path or default)
    docs_path = config.get("data", {}).get("insurance_path", "data/insurance")
    docs = load_docs_from_path(docs_path)
    sparse = SparseRetriever(docs)

    # Hybrid retriever
    hybrid = HybridRetriever(dense, sparse)

    # RouterAgent connects all agents
    model_name = config.get("llm", {}).get("model", "gpt-4o-mini")
    return RouterAgent(retriever=hybrid, faiss_indexer=faiss_indexer, model_name=model_name)


def main():
    router = build_router()

    def process_query(q: str) -> str:
        return asyncio.run(router.handle(q))

    chat = ConsoleChat(process_query)
    chat.start()


if __name__ == "__main__":
    main()

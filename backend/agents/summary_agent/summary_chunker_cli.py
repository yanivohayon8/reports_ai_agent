import argparse
import os
import glob
from pathlib import Path

import sys
import os

# Add the backend directory to Python path
backend_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, backend_path)

from indexer.indexer import FAISSIndexer
from backend.agents.summary_agent.summary_chunker import SummaryChunker
from backend.core.api_utils import get_openai_embeddings
from backend.core.config_utils import load_config
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main():
    parser = argparse.ArgumentParser(description="Summary Chunker CLI")
    parser.add_argument("--pdfs-directory", type=str, help="Path to the directory containing PDF files to index. Example: 'backend/data'")
    parser.add_argument("--pdf-path", type=str, help="Path to a single PDF file to index. Example: 'backend/data/report.pdf'")
    parser.add_argument("--config-path", type=str, default="backend/agents/summary_agent/summary_chunker_config.yaml")

    args = parser.parse_args()

    # Validate that either pdf_path or directory is provided, but not both
    if args.pdf_path is None and args.pdfs_directory is None:
        raise ValueError("Either --pdf-path or --pdfs-directory must be provided")
    
    if args.pdf_path is not None and args.pdfs_directory is not None:
        raise ValueError("Cannot specify both --pdf-path and --pdfs-directory. Use one or the other.")

    config = load_config(args.config_path)

    faiss_indexer_directory = Path(config["FAISSIndexer"]["directory_path"])
    if not faiss_indexer_directory.exists():
        os.makedirs(faiss_indexer_directory)

    embeddings = get_openai_embeddings(model=config["FAISSIndexer"]["embeddings"]["model"])    
    faiss_indexer = FAISSIndexer(embeddings, directory_path=faiss_indexer_directory)

    chunk_cfg = config.get("chunker", {})
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=int(chunk_cfg.get("chunk_size", 1000)),
        chunk_overlap=int(chunk_cfg.get("chunk_overlap", 200)),
        separators=chunk_cfg.get("separators", ["\n\n", "\n", " ", ""]) 
    )

    chunker = SummaryChunker(faiss_indexer, text_splitter)

    if args.pdf_path is not None:
        process_pdf(Path(args.pdf_path), chunker)
        
    elif args.pdfs_directory is not None:
        process_directory(Path(args.pdfs_directory), chunker)
    
    chunker.save(faiss_indexer_directory)
    print("Indexing completed successfully!")


def process_pdf(pdf_path: Path, chunker: SummaryChunker):
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")
        
    print(f"Processing single PDF: {pdf_path}")
    chunker.chunk(pdf_path)


def process_directory(directory_path: Path, chunker: SummaryChunker):
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found at {directory_path}")
        
    if not directory_path.is_dir():
        raise ValueError(f"Path {directory_path} is not a directory")
    
    # Find all PDF files in the directory using glob
    pdf_pattern = str(directory_path / "*.pdf")
    pdf_files = glob.glob(pdf_pattern)

    print(f"Processing {len(pdf_files)} PDF files in directory: {directory_path}")
    
    for pdf_path in pdf_files:
        process_pdf(Path(pdf_path), chunker)


if __name__ == "__main__":
    main()

import argparse
import os
import glob
from pathlib import Path

import sys
sys.path.append(".")

from indexer.indexer import FAISSIndexer
from agents.tableQA_agent.tableQA_chunker import TableQAChunker
from core.api_utils import get_openai_embeddings,get_llm_langchain_openai
from core.config_utils import load_config

def main():
    parser = argparse.ArgumentParser(description="TableQA Chunker CLI")
    parser.add_argument("--pdfs-directory", type=str, help="Path to the directory containing PDF files to index. Example: 'data'")
    parser.add_argument("--pdf-path", type=str, help="Path to a single PDF file to index. Example: 'data/report.pdf'")
    parser.add_argument("--config-path", type=str, default="agents/tableQA_agent/tableQA_chunker_config.yaml")

    args = parser.parse_args()

    # Validate that either pdf_path or directory is provided, but not both
    if args.pdf_path is None and args.pdfs_directory is None:
        raise ValueError("Either --pdf-path or --directory must be provided")
    
    if args.pdf_path is not None and args.pdfs_directory is not None:
        raise ValueError("Cannot specify both --pdf-path and --directory. Use one or the other.")

    config = load_config(args.config_path)

    faiss_indexer_directory = Path(config["FAISSIndexer"]["directory_path"])
    if not faiss_indexer_directory.exists():
        os.makedirs(faiss_indexer_directory)

    embeddings = get_openai_embeddings(model=config["FAISSIndexer"]["embeddings"]["model"],
                                       dimensions=config["FAISSIndexer"]["embeddings"]["dimensions"])
    faiss_indexer = FAISSIndexer(embeddings,directory_path=faiss_indexer_directory)
    llm = get_llm_langchain_openai(model=config["llm"]["model"])
    chunker = TableQAChunker(faiss_indexer,llm)

    if args.pdf_path is not None:
        process_pdf(Path(args.pdf_path),chunker)
        
    elif args.pdfs_directory is not None:
        # Handle directory of PDF files
        process_directory(Path(args.pdfs_directory),chunker)
    
    chunker.save(faiss_indexer_directory)
    print("Indexing completed successfully!")


def process_pdf(pdf_path:Path,chunker:TableQAChunker):
    if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")
        
    print(f"Processing single PDF: {pdf_path}")
    chunker.chunk(pdf_path)

def process_directory(directory_path:Path,chunker:TableQAChunker):
    if not directory_path.exists():
        raise FileNotFoundError(f"Directory not found at {directory_path}")
        
    if not directory_path.is_dir():
        raise ValueError(f"Path {directory_path} is not a directory")
    
    # Find all PDF files in the directory using glob
    pdf_pattern = str(directory_path / "*.pdf")
    pdf_files = glob.glob(pdf_pattern)

    print(f"Processing {len(pdf_files)} PDF files in directory: {directory_path}")
    
    for pdf_path in pdf_files:
        process_pdf(Path(pdf_path),chunker)

if __name__ == "__main__":
    main()
    

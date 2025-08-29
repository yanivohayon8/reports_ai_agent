import argparse
import os
import glob
from pathlib import Path

import sys
import os

# Add the backend directory to Python path
backend_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, backend_path)

from backend.core.text_splitter import get_text_splitter
from indexer import TextChunkerDeprecated,FAISSIndexer
from backend.indexer.graph_indexer import GraphIndexer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexer CLI")
    parser.add_argument("--directory", type=str, help="Path to the directory containing PDF files to index. Example: 'data'")
    parser.add_argument("--pdf-path", type=str, help="Path to a single PDF file to index. Example: 'data/report.pdf'")
    parser.add_argument("--faiss-indexer-directory", type=str, required=True, help="Path to the FAISS indexer directory. Example: 'vectordb_indexes/faiss_indexer'")
    parser.add_argument("--build-graph", action="store_true", help="Also build/update Graph RAG from the same docs")
    parser.add_argument("--graph-index-name", type=str, default="default", help="Name for the graph child index")

    args = parser.parse_args()

    # Validate that either pdf_path or directory is provided, but not both
    if args.pdf_path is None and args.directory is None:
        raise ValueError("Either --pdf-path or --directory must be provided")
    
    if args.pdf_path is not None and args.directory is not None:
        raise ValueError("Cannot specify both --pdf-path and --directory. Use one or the other.")

    faiss_indexer_directory = Path(args.faiss_indexer_directory)
    if not faiss_indexer_directory.exists():
        os.makedirs(faiss_indexer_directory)

    faiss_indexer = FAISSIndexer.from_small_embedding(directory_path=faiss_indexer_directory)
    text_splitter = get_text_splitter()
    text_chunker = TextChunkerDeprecated(faiss_indexer,text_splitter)

    collected_chunks = []

    if args.pdf_path is not None:
        # Handle single PDF file
        pdf_path = Path(args.pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")
        
        print(f"Processing single PDF: {pdf_path}")
        text_chunker.chunk(pdf_path)
        # For graph ingestion, reuse FAISS chunks
        from backend.core.pdf_reader import read_pdf
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n","\n"," ",""])
        pages = read_pdf(pdf_path, format="documents")
        collected_chunks.extend(splitter.split_documents(pages))
    
    elif args.directory is not None:
        # Handle directory of PDF files
        directory_path = Path(args.directory)
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found at {directory_path}")
        
        if not directory_path.is_dir():
            raise ValueError(f"Path {directory_path} is not a directory")
        
        # Find all PDF files in the directory using glob
        pdf_pattern = str(directory_path / "*.pdf")
        pdf_files = glob.glob(pdf_pattern)
        
        if not pdf_files:
            print(f"No PDF files found in directory: {directory_path}")
            exit(0)
        
        print(f"Found {len(pdf_files)} PDF files in {directory_path}")
        
        # Process each PDF file
        for pdf_file in pdf_files:
            pdf_path = Path(pdf_file)
            print(f"Processing PDF: {pdf_path}")
            text_chunker.chunk(pdf_path)
            from backend.core.pdf_reader import read_pdf
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, separators=["\n\n","\n"," ",""])
            pages = read_pdf(pdf_path, format="documents")
            collected_chunks.extend(splitter.split_documents(pages))
    
    text_chunker.save(faiss_indexer_directory)
    print("Indexing completed successfully!")

    if args.build_graph:
        try:
            graph = GraphIndexer()
            graph.add_langchain_documents(collected_chunks, args.graph_index_name)
            graph.build_graph()
            print("Graph RAG built successfully")
        except Exception as e:
            print(f"Graph RAG build failed: {e}")
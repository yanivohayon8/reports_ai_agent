import argparse
import os
import glob
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter

from indexer import TextChunker,FAISSIndexer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexer CLI")
    parser.add_argument("--directory", type=str, help="Path to the directory containing PDF files to index. Example: 'pdfs'")
    parser.add_argument("--pdf-path", type=str, help="Path to a single PDF file to index. Example: 'pdfs/report.pdf'")
    parser.add_argument("--faiss-indexer-directory", type=str, required=True, help="Path to the FAISS indexer directory. Example: 'vectordb_indexes/faiss_indexer'")
    parser.add_argument("--chunk-size", type=int, default=300, help="Chunk size for the text splitter. Example: 300")
    parser.add_argument("--chunk-overlap", type=int, default=50, help="Chunk overlap for the text splitter. Example: 50")

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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    text_chunker = TextChunker(faiss_indexer,text_splitter)

    if args.pdf_path is not None:
        # Handle single PDF file
        pdf_path = Path(args.pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at {pdf_path}")
        
        print(f"Processing single PDF: {pdf_path}")
        text_chunker.chunk(pdf_path)
    
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
    
    text_chunker.save(faiss_indexer_directory)
    print("Indexing completed successfully!")
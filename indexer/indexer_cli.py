import argparse
import os
from pathlib import Path

from indexer import TextChunker,FAISSIndexer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indexer CLI")
    parser.add_argument("--pdf-path",type=str,required=True,help="Path to the PDF file to index. Example: 'pdfs/report.pdf'")
    parser.add_argument("--faiss-indexer-directory",type=str,required=True,help="Path to the FAISS indexer directory. Example: 'vectordb_indexes/faiss_indexer'")

    args = parser.parse_args()

    faiss_indexer_directory = Path(args.faiss_indexer_directory)
    if not faiss_indexer_directory.exists():
        os.makedirs(faiss_indexer_directory)

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")

    faiss_indexer = FAISSIndexer.from_small_embedding(directory_path=faiss_indexer_directory)
    text_chunker = TextChunker(faiss_indexer,chunk_size=300,chunk_overlap=50)

    text_chunker.chunk(pdf_path)
    text_chunker.save(faiss_indexer_directory)
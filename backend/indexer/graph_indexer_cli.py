import argparse
import os
import sys
import glob
from pathlib import Path

# הוספת project root ל-PYTHONPATH
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
os.chdir(project_root)

from backend.indexer.graph_indexer import GraphIndexer
from backend.core.pdf_reader import read_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


def collect_chunks_from_pdf(pdf_path: Path, chunk_size: int = 1000, chunk_overlap: int = 200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    pages = read_pdf(pdf_path, format="documents")
    return splitter.split_documents(pages)


def main():
    parser = argparse.ArgumentParser(description="Graph Indexer CLI")
    parser.add_argument("--directory", type=str, help="Directory containing PDF files")
    parser.add_argument("--pdf-path", type=str, help="Single PDF file path")
    parser.add_argument("--graph-index-name", type=str, default="default", help="Name for the child index")
    parser.add_argument("--save-path", type=str, default=None, help="Save the built graph to this path")
    parser.add_argument("--load-path", type=str, default=None, help="Load an existing graph from this path")
    parser.add_argument("--query", type=str, default=None, help="Run a query on the graph after loading")

    args = parser.parse_args()

    # --- Load mode ---
    if args.load_path:
        graph = GraphIndexer.load(args.load_path)
        print(f"Graph loaded ✓ from {args.load_path}")
        if args.query:
            result = graph.retrieve(args.query)

            print(f"Query: {args.query}")
            print("Answer:", result["answer"])

            if result["sources"]:
                print("Sources:")
                for s in result["sources"]:
                    print(f" - {s['source']} (page {s['page']}): {s['snippet']}")
        return

    # --- Build mode ---
    if args.pdf_path is None and args.directory is None:
        raise ValueError("Either --pdf-path or --directory must be provided")
    if args.pdf_path is not None and args.directory is not None:
        raise ValueError("Use either --pdf-path or --directory, not both")

    graph = GraphIndexer()

    if args.pdf_path:
        pdf_path = Path(args.pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        print(f"GraphIndexer: collecting chunks from {pdf_path}")
        chunks = collect_chunks_from_pdf(pdf_path)
        print(f"GraphIndexer: adding {len(chunks)} chunks to index '{args.graph_index_name}'")
        graph.add_langchain_documents(chunks, args.graph_index_name)

    else:
        directory_path = Path(args.directory)
        if not directory_path.exists() or not directory_path.is_dir():
            raise FileNotFoundError(f"Directory not found or not a dir: {directory_path}")
        pdf_files = glob.glob(str(directory_path / "*.pdf"))
        if not pdf_files:
            print(f"No PDFs found in {directory_path}")
            sys.exit(0)
        for pdf_file in pdf_files:
            pdf_path = Path(pdf_file)
            idx_name = f"{args.graph_index_name}:{pdf_path.stem}"
            print(f"GraphIndexer: collecting chunks from {pdf_path}")
            chunks = collect_chunks_from_pdf(pdf_path)
            print(f"GraphIndexer: adding {len(chunks)} chunks to index '{idx_name}'")
            graph.add_langchain_documents(chunks, idx_name)

    print("GraphIndexer: building composable graph...")
    graph.build_graph()
    used = graph.get_used_input()
    print(f"Graph built ✓ | nodes={used.get('nodes')} | indices={used.get('indices')}")

    # Save if requested
    if args.save_path:
        graph.save(args.save_path)
        print(f"GraphIndexer: graph saved to {args.save_path}")


if __name__ == "__main__":
    main()

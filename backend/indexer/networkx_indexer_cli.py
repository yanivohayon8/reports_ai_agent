import argparse
import os
import sys
import glob
import networkx as nx
from pathlib import Path
import matplotlib.pyplot as plt
import pickle

# Add project root for imports
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)
os.chdir(project_root)

from backend.core.pdf_reader import read_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.indexer.networkx_indexer import NetworkXIndexer


def collect_chunks_from_pdf(pdf_path: Path, chunk_size: int = 800, chunk_overlap: int = 100):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    pages = read_pdf(pdf_path, format="documents")
    return splitter.split_documents(pages)


def visualize_graph(g: nx.Graph, out_path: str = None, show: bool = True,
                    focus_node: str = None, depth: int = 1):
    """
    Visualize the graph (whole graph or subgraph around a node).
    """
    if focus_node and focus_node in g:
        from networkx import single_source_shortest_path_length
        sub_nodes = single_source_shortest_path_length(g, focus_node, cutoff=depth).keys()
        sg = g.subgraph(sub_nodes).copy()
        print(f"[Visualizer] Showing subgraph of {focus_node} with {len(sg)} nodes")
    else:
        sg = g

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(sg, seed=42)

    nx.draw_networkx_nodes(sg, pos, node_size=80, node_color="skyblue")
    nx.draw_networkx_edges(sg, pos, alpha=0.5, edge_color="gray")

    labels = {n: n for n in list(sg.nodes)[:30]}
    nx.draw_networkx_labels(sg, pos, labels=labels, font_size=7)

    plt.axis("off")
    if out_path:
        plt.savefig(out_path, dpi=200)
        print(f"Graph visualization saved to {out_path}")
    if show:
        plt.show()
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="NetworkX Graph Builder CLI (with visualization)")
    parser.add_argument("--directory", type=str, help="Directory containing PDF files")
    parser.add_argument("--pdf-path", type=str, help="Single PDF file path")
    parser.add_argument("--graph-name", type=str, default="nx_graph", help="Name of the graph")
    parser.add_argument("--save-path", type=str, default=None, help="Path to save the graph (.graphml or .gpickle)")
    parser.add_argument("--load-path", type=str, default=None, help="Load an existing graph")
    parser.add_argument("--query-node", type=str, default=None, help="Query: get neighbors of this node id")
    parser.add_argument("--depth", type=int, default=1, help="Depth for neighbor query")
    parser.add_argument("--visualize", action="store_true", help="Visualize the graph after build/load")
    parser.add_argument("--visualize-path", type=str, default=None, help="Save visualization to this file")

    args = parser.parse_args()

    # --- Load mode ---
    if args.load_path:
        ext = Path(args.load_path).suffix
        if ext == ".graphml":
            g = nx.read_graphml(args.load_path)
        else:
            with open(args.load_path, "rb") as f:
                g = pickle.load(f)
        print(f"Graph loaded ✓ from {args.load_path} | nodes={len(g)} edges={g.number_of_edges()}")

        if args.query_node:
            if args.query_node in g:
                from networkx import single_source_shortest_path_length
                sub_nodes = single_source_shortest_path_length(g, args.query_node, cutoff=args.depth).keys()
                texts = [g.nodes[n].get("text", "") for n in sub_nodes if "text" in g.nodes[n]]
                print(f"Neighbors of {args.query_node} (depth={args.depth}):")
                for t in texts[:5]:
                    print(" -", t[:200], "...")
            else:
                print(f"Node {args.query_node} not found in graph")

        if args.visualize:
            visualize_graph(g, out_path=args.visualize_path, show=True,
                            focus_node=args.query_node, depth=args.depth)
        return

    # --- Build mode ---
    if args.pdf_path is None and args.directory is None:
        raise ValueError("Either --pdf-path or --directory must be provided")
    if args.pdf_path is not None and args.directory is not None:
        raise ValueError("Use either --pdf-path or --directory, not both")

    nx_indexer = NetworkXIndexer()

    if args.pdf_path:
        pdf_path = Path(args.pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        chunks = collect_chunks_from_pdf(pdf_path)
        nx_indexer.add_documents(chunks, source_name=pdf_path.stem)

    else:
        pdf_files = glob.glob(str(Path(args.directory) / "*.pdf"))
        if not pdf_files:
            print(f"No PDFs found in {args.directory}")
            sys.exit(0)
        for pdf_file in pdf_files:
            pdf_path = Path(pdf_file)
            chunks = collect_chunks_from_pdf(pdf_path)
            nx_indexer.add_documents(chunks, source_name=pdf_path.stem)

    g = nx_indexer.graph
    print(f"Graph built ✓ | nodes={len(g)} edges={g.number_of_edges()}")

    if args.save_path:
        ext = Path(args.save_path).suffix
        if ext == ".graphml":
            nx.write_graphml(g, args.save_path)
        else:
            with open(args.save_path, "wb") as f:
                pickle.dump(g, f)
        print(f"Graph saved to {args.save_path}")

    if args.visualize:
        visualize_graph(g, out_path=args.visualize_path, show=True,
                        focus_node=args.query_node, depth=args.depth)


if __name__ == "__main__":
    main()

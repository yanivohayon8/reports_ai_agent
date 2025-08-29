import argparse
from pathlib import Path
import sys

sys.path.append(".")

from backend.core.config_utils import load_config
from backend.agents.router_agent.router_dataset_synthesizer import (
    create_router_dataset_from_pdf,
    save_router_dataset,
)


def main():
    parser = argparse.ArgumentParser(description="Router dataset synthesizer CLI")
    parser.add_argument("--pdf-path", type=str, help="Path to a single PDF")
    parser.add_argument("--pdfs-directory", type=str, help="Directory with PDFs")
    parser.add_argument("--config-path", type=str, default="backend/agents/router_agent/router_dataset_synthesizer_config.yaml")
    parser.add_argument("--n-chunks", type=int, default=10)
    args = parser.parse_args()

    if not args.pdf_path and not args.pdfs_directory:
        raise ValueError("Provide --pdf-path or --pdfs-directory")
    if args.pdf_path and args.pdfs_directory:
        raise ValueError("Use only one of --pdf-path or --pdfs-directory")

    cfg = load_config(args.config_path)
    chunk_size = cfg["chunker"]["chunk_size"]
    chunk_overlap = cfg["chunker"]["chunk_overlap"]
    out_dir = Path(cfg["output"]["dir"]).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    def process(pdf: Path):
        rows = create_router_dataset_from_pdf(pdf, chunk_size=chunk_size, chunk_overlap=chunk_overlap, n_chunks=args.n_chunks)
        out = save_router_dataset(rows, out_dir, pdf.stem)
        print(f"✓ Saved {out}")

    if args.pdf_path:
        process(Path(args.pdf_path))
    else:
        for p in Path(args.pdfs_directory).glob("*.pdf"):
            process(p)


if __name__ == "__main__":
    main()



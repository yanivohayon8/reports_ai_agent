import argparse
from pathlib import Path
import sys
sys.path.append(".")

from backend.core.config_utils import load_config
from backend.core.api_utils import get_llm_langchain_openai
from backend.agents.tableQA_agent.tableQA_dataset_synthesizer import (
    TableQADatasetSynthesizer,
    save_tableQA_dataset,
)


def main():
    parser = argparse.ArgumentParser(description="TableQA dataset synthesizer CLI")
    parser.add_argument("--pdf-path", type=str, help="Path to a single PDF")
    parser.add_argument("--pdfs-directory", type=str, help="Directory with PDFs")
    parser.add_argument("--config-path", type=str, default="backend/agents/tableQA_agent/tableQA_dataset_synthesizer_config.yaml")
    parser.add_argument("--n-chunks", type=int, default=10, help="How many chunks to sample per PDF")
    args = parser.parse_args()

    if not args.pdf_path and not args.pdfs_directory:
        raise ValueError("Provide --pdf-path or --pdfs-directory")
    if args.pdf_path and args.pdfs_directory:
        raise ValueError("Use only one of --pdf-path or --pdfs-directory")

    cfg = load_config(args.config_path)
    llm = get_llm_langchain_openai(model=cfg["llm"]["model"])
    synth = TableQADatasetSynthesizer(
        llm=llm,
        min_chunk_size=cfg["chunker"]["min_chunk_size"],
        max_chunk_size=cfg["chunker"]["max_chunk_size"],
        min_chunk_overlap=cfg["chunker"]["min_chunk_overlap"],
        max_chunk_overlap=cfg["chunker"]["max_chunk_overlap"],
    )

    output_dir = Path(cfg["output"]["dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    def process(pdf: Path):
        rows = synth.create_dataset(pdf, n_chunks=args.n_chunks)
        save_tableQA_dataset(rows, pdf, output_dir)
        print(f"✓ Saved dataset for {pdf.name}")

    if args.pdf_path:
        process(Path(args.pdf_path))
    else:
        for pdf in Path(args.pdfs_directory).glob("*.pdf"):
            process(pdf)


if __name__ == "__main__":
    main()

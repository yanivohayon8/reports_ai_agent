"""Command-line interface for timeline summarizer."""
#!/usr/bin/env python
import argparse
import pathlib
from dotenv import load_dotenv

from timeline.io_utils import read_file
from timeline.summarizer import TimelineSummarizer

load_dotenv()

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("timeline-cli")
    p.add_argument("file", type=pathlib.Path, help="Input TXT or PDF")
    p.add_argument("--method", choices=["map_reduce", "refine"], default="map_reduce")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--chunk-size", type=int, default=1500)
    p.add_argument("--chunk-overlap", type=int, default=200)
    p.add_argument("--token-based", action="store_true")
    return p.parse_args()

def main() -> None:
    args = parse_args()
    raw_text = read_file(args.file)
    if not raw_text:
        raise ValueError("File could not be decoded or is empty.")

    summarizer = TimelineSummarizer(
        model_name=args.model,
        temperature=args.temperature,
        method=args.method,  # type: ignore[arg-type]
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        token_based_split=args.token_based,
    )
    print("\n=== TIMELINE SUMMARY ===\n")
    print(summarizer.summarize(raw_text))

if __name__ == "__main__":
    main()

"""CLI entry point for QnA chat."""
import argparse
import pathlib
from dotenv import load_dotenv

from qna.qna_core import build_qna_tool


load_dotenv()

def parse_args():
    p = argparse.ArgumentParser("timeline-qna-cli")
    p.add_argument("file", type=pathlib.Path, help="Input TXT or PDF")
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-k", type=int, default=4)
    p.add_argument("--score-threshold", type=float)
    p.add_argument("--token-based", action="store_true")
    p.add_argument("--chunk-size", type=int, default=1500)
    p.add_argument("--chunk-overlap", type=int, default=200)
    p.add_argument("--persist", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    tool = build_qna_tool(
        args.file,
        model=args.model,
        temperature=args.temperature,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        token_based=args.token_based,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        persist=args.persist,
    )

    print("\n=== QnA Chat === (type 'exit' to quit)\n")
    while True:
        try:
            q = input("You: ")
        except EOFError:
            break
        if q.strip().lower() in {"exit", "quit"}:
            break
        print("\n" + tool.run(q) + "\n")

if __name__ == "__main__":
    main()

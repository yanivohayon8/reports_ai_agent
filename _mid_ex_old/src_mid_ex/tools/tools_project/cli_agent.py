#!/usr/bin/env python
"""
cli_agent.py — CLI for HaystackAgent: routes between QnA and timeline summarizer.

Usage examples:
    # Single factual question:
    python cli_agent.py report.pdf --question "When did the event happen?"

    # Interactive chat (QnA or timeline as needed):
    python cli_agent.py report.pdf --chat
"""

from __future__ import annotations
import argparse
import pathlib
from typing import Literal

from dotenv import load_dotenv
from agent.core import HaystackAgent

load_dotenv()


def parse_args(args=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser("cli_agent")
    # Input document
    parser.add_argument(
        "file", type=pathlib.Path, help="Path to input TXT or PDF document"
    )
    # Modes: single question or interactive chat
    parser.add_argument(
        "-q", "--question", type=str, help="Ask one question and exit"
    )
    parser.add_argument(
        "--chat", action="store_true", help="Interactive chat mode"
    )
    # LLM parameters
    parser.add_argument(
        "--model", default="gpt-4o-mini", help="OpenAI model name"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Sampling temperature"
    )
    # Timeline summarizer parameters
    parser.add_argument(
        "--method",
        choices=["map_reduce", "refine"],
        default="map_reduce",
        help="Timeline summarization strategy",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1500,
        help="Chunk size for summarization",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks",
    )
    parser.add_argument(
        "--token-based",
        action="store_true",
        help="Split summarization by token count instead of characters",
    )
    # QnA parameters
    parser.add_argument(
        "--top-k",
        type=int,
        default=4,
        help="Number of documents to retrieve per question",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        help="Minimum retrieval score threshold",
    )
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Persist FAISS index to disk for faster subsequent runs",
    )
    return parser.parse_args(args)


def main() -> None:
    args = parse_args()

    # Initialize the agent with both QnA and timeline summarizer
    agent = HaystackAgent(
        model_name=args.model,
        temperature=args.temperature,
        summary_method=args.method,  # type: ignore[arg-type]
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        token_based=args.token_based,
        top_k=args.top_k,
        score_threshold=args.score_threshold,
        persist=args.persist,
    )

    # Load the document and build internal tools
    agent.load_text(args.file)

    # Single-question mode
    if args.question:
        print("\n=== ANSWER ===\n")
        print(agent.answer(args.question))
        return

    # Interactive chat mode
    if args.chat:
        print("\n[Interactive chat — type 'exit' or 'quit' to leave]\n")
        while True:
            try:
                user_q = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nGoodbye!")
                break
            if user_q.lower() in {"exit", "quit"}:
                print("Goodbye!")
                break
            response = agent.answer(user_q)
            print("\n" + response + "\n")
        return

    # If neither --question nor --chat was specified, show usage
    print("Error: please provide either --question or --chat")
    print("Use -h for help.")


if __name__ == "__main__":
    main()

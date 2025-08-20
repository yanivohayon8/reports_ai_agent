import argparse
import sys
from  pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from core.api_utils import get_llm_langchain_openai
from agents.summary_agent.summary import SummaryAgent
from core.text_splitter import get_text_splitter    

def main():
    parser = argparse.ArgumentParser(description="Summarize a PDF file")
    parser.add_argument("pdf_path", type=str, help="The path to the PDF file")
    parser.add_argument("--method", type=str, help="The method to use for summarization")
    parser.add_argument("--model", type=str, help="The model to use for summarization", default="gpt-4o-mini")
    args = parser.parse_args()

    text_splitter = get_text_splitter()
    llm = get_llm_langchain_openai(model=args.model)
    summary_agent = SummaryAgent(text_splitter,llm)
    summary = summary_agent.summarize_single_pdf(Path(args.pdf_path), args.method)
    print(summary)

if __name__ == "__main__":
    main()




import argparse

from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.api_utils import get_llm_langchain_openai
from summary_agent.summary import SummaryAgent
from  pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Summarize a PDF file")
    parser.add_argument("pdf_path", type=str, help="The path to the PDF file")
    parser.add_argument("method", type=str, help="The method to use for summarization")
    args = parser.parse_args()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50)
    llm = get_llm_langchain_openai(model="gpt-4o-mini")
    summary_agent = SummaryAgent(text_splitter,llm)
    summary = summary_agent.summarize_single_pdf(Path(args.pdf_path), args.method)
    print(summary)

if __name__ == "__main__":
    main()




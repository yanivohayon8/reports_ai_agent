import sys
import os

# Add the backend directory to Python path
backend_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, backend_path)

from core.api_utils import get_llm_langchain_openai
from pathlib import Path
from agents.needle_agent.needle_dataset_synthesizer import DatasetSynthesizer, save_needle_dataset
import argparse
from core.config_utils import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-path", type=str)
    parser.add_argument("--pdf-directory", type=str)
    parser.add_argument("--config-path", type=str, default="backend/agents/needle_agent/needle_dataset_synthesizer_config.yaml")
    parser.add_argument("--output-path", type=str, help="Output path for the dataset. If not provided, will use timestamped filename in data/evaluation_datasets/needle_agent/")
    args = parser.parse_args()

    config = load_config(args.config_path)

    llm = get_llm_langchain_openai(model=config["model"])
    dataset_creator = DatasetSynthesizer(llm,
                                     config["min_chunk_size"],config["max_chunk_size"],
                                     config["min_chunk_overlap"],config["max_chunk_overlap"])    

    # Generate output directory if not provided
    if not args.output_path:
        output_dir = Path("data/evaluation_datasets/needle_agent")
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = Path(args.output_path)
        if output_dir.suffix:  # If it's a file path, use its parent directory
            output_dir = output_dir.parent
        output_dir.mkdir(parents=True, exist_ok=True)

    total_qa_pairs = 0
    
    if args.pdf_path:
        # Single PDF file
        pdf_path = Path(args.pdf_path)
        questions_and_answers = dataset_creator.create_dataset(pdf_path, n_chunks=config["n_chunks"])
        
        total_qa_pairs += save_needle_dataset(questions_and_answers, pdf_path, output_dir)

    elif args.pdf_directory:
        # Multiple PDF files
        pdf_paths = list(Path(args.pdf_directory).glob("*.pdf"))
        
        if not pdf_paths:
            print(f"No PDF files found in directory: {args.pdf_directory}")
            sys.exit(1)
        
        print(f"Found {len(pdf_paths)} PDF files to process")
        
        for pdf_path in pdf_paths:
            try:
                questions_and_answers = dataset_creator.create_dataset(pdf_path, n_chunks=config["n_chunks"])
                total_qa_pairs += save_needle_dataset(questions_and_answers, pdf_path, output_dir)
                
            except Exception as e:
                print(f"Error processing {pdf_path.name}: {e}")
                continue
    else:
        raise ValueError("Either pdf_path or pdf_directory must be provided")
    
    print(f"\nTotal processing complete: {total_qa_pairs} question-answer pairs across all files")
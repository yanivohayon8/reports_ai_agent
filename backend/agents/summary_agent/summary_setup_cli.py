#!/usr/bin/env python3
"""
Summary Agent Setup CLI
Processes insurance PDFs and builds the FAISS index for the summary agent.
"""

import asyncio
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.core.api_utils import get_llm_langchain_openai
from backend.agents.summary_agent.summary_chunker import SummaryChunker
from backend.indexer.indexer import FAISSIndexer
from backend.core.pdf_reader import read_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter


def setup_summary_agent():
    """Set up the summary agent with insurance data files."""
    print("🚀 Setting up Summary Agent with Insurance Data...")
    
    # Configuration
    insurance_data_dir = Path("backend/data/insurance")
    summary_index_dir = Path("backend/vectordb_indexes/summary_insurance")
    
    # Create index directory if it doesn't exist
    summary_index_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Using index directory: {summary_index_dir}")
    
    # Initialize FAISS indexer
    try:
        print("🔧 Initializing FAISS indexer...")
        indexer = FAISSIndexer.from_small_embedding(directory_path=str(summary_index_dir))
        print("✅ FAISS indexer initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize FAISS indexer: {e}")
        return None
    
    # Initialize text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    
    # Initialize summary chunker
    chunker = SummaryChunker(faiss_indexer=indexer, text_splitter=text_splitter)
    
    # Process insurance PDFs
    pdf_files = list(insurance_data_dir.glob("*.pdf"))
    if not pdf_files:
        print("❌ No PDF files found in insurance data directory")
        return None
    
    print(f"📚 Found {len(pdf_files)} PDF files to process:")
    for pdf_file in pdf_files:
        print(f"   - {pdf_file.name}")
    
    # Process each PDF
    processed_count = 0
    for pdf_file in pdf_files:
        try:
            print(f"\n📖 Processing: {pdf_file.name}")
            
            # Check if PDF is already processed
            if indexer.is_pdf_processed(pdf_file):
                print(f"   ⏭️  Already processed, skipping...")
                continue
            
            # Process the PDF
            chunks = chunker.chunk(pdf_file)
            print(f"   ✅ Created {len(chunks)} chunks")
            processed_count += 1
            
        except Exception as e:
            print(f"   ❌ Error processing {pdf_file.name}: {e}")
            continue
    
    print(f"\n🎉 Summary Agent setup complete!")
    print(f"   📊 Processed {processed_count} PDF files")
    print(f"   🗂️  Index stored in: {summary_index_dir}")
    
    return chunker


def test_summary_agent(chunker):
    """Test the summary agent with a sample query."""
    if not chunker:
        print("❌ No chunker available for testing")
        return
    
    print("\n🧪 Testing Summary Agent...")
    
    # Initialize LLM
    try:
        llm = get_llm_langchain_openai(model="gpt-4o-mini")
        print("✅ LLM initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize LLM: {e}")
        return
    
    # Test query
    test_query = "What are the main types of insurance policies available?"
    
    try:
        # Get relevant documents
        relevant_docs = chunker.faiss_indexer.retrieve(test_query, num_documents=5)
        if relevant_docs:
            print(f"✅ Found {len(relevant_docs)} relevant document chunks")
            print(f"📝 Sample chunk content: {relevant_docs[0].page_content[:200]}...")
        else:
            print("❌ No relevant documents found")
    except Exception as e:
        print(f"❌ Error testing summary agent: {e}")


def main():
    """Main CLI function."""
    print("=" * 60)
    print("🔍 SUMMARY AGENT SETUP TOOL")
    print("=" * 60)
    
    # Setup summary agent
    chunker = setup_summary_agent()
    
    if chunker:
        # Test the setup
        test_summary_agent(chunker)
        
        print("\n" + "=" * 60)
        print("🎯 NEXT STEPS:")
        print("1. The summary agent is now configured with your insurance data")
        print("2. You can use it through the router agent in the web API")
        print("3. Try asking questions like:")
        print("   - 'Summarize the tour and care policies'")
        print("   - 'What are the key features of Menora policies?'")
        print("   - 'Give me an overview of the insurance coverage'")
        print("=" * 60)
    else:
        print("\n❌ Summary Agent setup failed. Please check the logs above.")


if __name__ == "__main__":
    main()

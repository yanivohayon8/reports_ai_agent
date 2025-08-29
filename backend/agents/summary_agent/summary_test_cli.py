#!/usr/bin/env python3
"""
Summary Agent Test CLI
Test the summary agent with queries about insurance data.
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
from backend.agents.summary_agent.summary_agent import SummaryAgent
from backend.agents.summary_agent.summary_chunker import SummaryChunker
from backend.indexer.indexer import FAISSIndexer
from langchain_text_splitters import RecursiveCharacterTextSplitter


async def test_summary_agent():
    """Test the summary agent with insurance data."""
    print("🧪 Testing Summary Agent with Insurance Data...")
    
    # Configuration
    summary_index_dir = Path("backend/vectordb_indexes/summary_insurance")
    
    if not summary_index_dir.exists():
        print("❌ Summary index directory not found. Please run summary_setup_cli.py first.")
        return
    
    try:
        # Initialize components
        print("🔧 Initializing components...")
        
        # FAISS indexer
        indexer = FAISSIndexer.from_small_embedding(directory_path=str(summary_index_dir))
        print("✅ FAISS indexer initialized")
        
        # Text splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        print("✅ Text splitter initialized")
        
        # Summary chunker
        chunker = SummaryChunker(faiss_indexer=indexer, text_splitter=text_splitter)
        print("✅ Summary chunker initialized")
        
        # LLM
        llm = get_llm_langchain_openai(model="gpt-4o-mini")
        print("✅ LLM initialized")
        
        # Summary agent
        summary_agent = SummaryAgent(chunker=chunker, llm=llm)
        print("✅ Summary agent initialized")
        
        # Test queries
        test_queries = [
            "What are the main types of insurance policies available?",
            "Summarize the tour and care policies",
            "What are the key features of Menora policies?",
            "Give me an overview of the insurance coverage"
        ]
        
        print("\n" + "=" * 60)
        print("🔍 TESTING SUMMARY AGENT")
        print("=" * 60)
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Test {i}: {query}")
            print("-" * 40)
            
            try:
                result = await summary_agent.handle(query)
                print(f"🤖 Agent: {result.get('agent', 'Unknown')}")
                print(f"💭 Reasoning: {result.get('reasoning', 'N/A')}")
                print(f"📄 Answer: {result.get('answer', 'No answer')}")
                print("✅ Success")
            except Exception as e:
                print(f"❌ Error: {e}")
        
        print("\n" + "=" * 60)
        print("🎉 Summary Agent Test Complete!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Failed to test summary agent: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Main function."""
    print("=" * 60)
    print("🔍 SUMMARY AGENT TEST TOOL")
    print("=" * 60)
    
    asyncio.run(test_summary_agent())


if __name__ == "__main__":
    main()

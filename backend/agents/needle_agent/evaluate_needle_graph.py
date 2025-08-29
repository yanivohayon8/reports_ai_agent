#!/usr/bin/env python3
"""
Simple CLI to evaluate NeedleAgentGraph (Graph RAG) performance.
Usage: python evaluate_needle_graph.py
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from needle_evaluator_graph import NeedleEvaluatorGraph
from backend.agents.router_agent.router import create_router_from_config
from backend.core.api_utils import get_llm_langchain_openai


def main():
    """Main CLI function."""
    
    print("🚀 NeedleAgentGraph (Graph RAG) Evaluation CLI")
    print("=" * 50)
    
    try:
        # Check if dataset exists
        dataset_path = Path("backend/data/evaluation_datasets/needle_agent/client3_report3_MenoraPolicy.jsonl")
        if not dataset_path.exists():
            print(f"❌ Dataset not found at: {dataset_path}")
            print("   Please ensure the dataset exists before running evaluation.")
            return False
        
        print(f"📊 Dataset: {dataset_path}")
        print(f"🎯 Output: needle_graph_evaluation_results.json")
        print()
        
        # Initialize LLM
        print("🔧 Initializing LLM...")
        llm = get_llm_langchain_openai(model="gpt-4o-mini")
        print("✅ LLM initialized: gpt-4o-mini")
        
        # Create router
        print("🔧 Creating router...")
        router = create_router_from_config()
        print("✅ Router created")
        
        # Check agent type
        agent_type = type(router.needle_agent).__name__
        print(f"🤖 Agent type: {agent_type}")
        
        if "Graph" not in agent_type:
            print("⚠️  Warning: Not using NeedleAgentGraph!")
            print("   Check your config.yaml - ensure NeedleAgent.type: graph and GraphRAG.enabled: true")
            print("   Continuing with current agent type...")
        else:
            print("🎯 Confirmed: Using NeedleAgentGraph (Graph RAG)")
        
        # Create evaluator
        print("🔧 Creating evaluator...")
        evaluator = NeedleEvaluatorGraph(dataset_path, llm)
        print("✅ Evaluator created")
        
        # Run evaluation
        print("\n📈 Running evaluation...")
        print("=" * 50)
        
        result = evaluator.evaluate(router.needle_agent)
        
        if result is None:
            print("❌ Evaluation failed!")
            return False
        
        # Display results
        print("\n📊 Evaluation Results:")
        print("-" * 30)
        
        if hasattr(result, 'to_pandas'):
            df_results = result.to_pandas()
            if not df_results.empty:
                metrics_row = df_results.iloc[0]
                print(f"Faithfulness: {metrics_row.get('faithfulness', 'N/A'):.4f}")
                print(f"Context Recall: {metrics_row.get('context_recall', 'N/A'):.4f}")
                print(f"LLM Context Precision: {metrics_row.get('llm_context_precision_with_reference', 'N/A'):.4f}")
            else:
                print("No metrics available in results")
        else:
            print(f"Raw result: {result}")
        
        # Save results
        output_path = Path("needle_graph_evaluation_results.json")
        print(f"\n💾 Saving results to {output_path}...")
        evaluator.save_results(result, output_path, router.needle_agent)
        print("✅ Results saved successfully!")
        
        print("\n🎉 Evaluation completed successfully!")
        return True
        
    except KeyboardInterrupt:
        print("\n⚠️  Evaluation interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Evaluation failed. Check the error messages above.")
        sys.exit(1)
    else:
        print("\n✅ Evaluation completed successfully!")
        sys.exit(0)

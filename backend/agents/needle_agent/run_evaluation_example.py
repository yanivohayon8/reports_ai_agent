#!/usr/bin/env python3
"""
Example script showing how to use NeedleEvaluatorGraph with configuration.
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.agents.needle_agent.needle_evaluator_graph import NeedleEvaluatorGraph
from backend.agents.router_agent.router import create_router_from_config
from backend.core.api_utils import get_llm_langchain_openai


def run_simple_evaluation():
    """Run a simple evaluation using NeedleEvaluatorGraph."""
    
    print("🚀 Running Simple NeedleAgentGraph Evaluation...")
    
    try:
        # Initialize LLM
        llm = get_llm_langchain_openai(model="gpt-4o-mini")
        print("✅ LLM initialized")
        
        # Create router (should use NeedleAgentGraph if configured)
        router = create_router_from_config()
        print("✅ Router created")
        
        # Check agent type
        agent_type = type(router.needle_agent).__name__
        print(f"🤖 Agent type: {agent_type}")
        
        if "Graph" not in agent_type:
            print("⚠️  Warning: Not using NeedleAgentGraph. Check your config.yaml")
            return False
        
        # Create evaluator
        dataset_path = Path("backend/data/evaluation_datasets/needle_agent/client3_report3_MenoraPolicy.jsonl")
        if not dataset_path.exists():
            print(f"❌ Dataset not found at {dataset_path}")
            return False
            
        evaluator = NeedleEvaluatorGraph(dataset_path, llm)
        print("✅ Evaluator created")
        
        # Run evaluation
        print("\n📈 Running evaluation...")
        result = evaluator.evaluate(router.needle_agent)
        
        # Display results
        print("\n📊 Evaluation Results:")
        
        # Handle different result formats from ragas
        if hasattr(result, 'to_pandas'):
            # Convert to pandas to extract metrics
            df_results = result.to_pandas()
            if not df_results.empty:
                # Get the first row which contains the metrics
                metrics_row = df_results.iloc[0]
                print(f"Faithfulness: {metrics_row.get('faithfulness', 'N/A'):.4f}")
                print(f"Context Recall: {metrics_row.get('context_recall', 'N/A'):.4f}")
                print(f"LLM Context Precision: {metrics_row.get('llm_context_precision_with_reference', 'N/A'):.4f}")
            else:
                print("No metrics available in results")
        elif hasattr(result, '__dict__'):
            # Try to access attributes directly
            print(f"Faithfulness: {getattr(result, 'faithfulness', 'N/A'):.4f}")
            print(f"Context Recall: {getattr(result, 'context_recall', 'N/A'):.4f}")
            print(f"LLM Context Precision: {getattr(result, 'llm_context_precision_with_reference', 'N/A'):.4f}")
        else:
            print(f"Raw result: {result}")
        
        # Save results
        output_path = "simple_evaluation_results.json"
        evaluator.save_results(result, Path(output_path), router.needle_agent)
        print(f"\n💾 Results saved to {output_path}")
        
        print("\n🎉 Simple evaluation completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_simple_evaluation()
    sys.exit(0 if success else 1)

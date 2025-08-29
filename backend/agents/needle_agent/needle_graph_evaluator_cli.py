#!/usr/bin/env python3
"""
CLI for evaluating NeedleAgent Graph RAG performance.
Usage: python needle_graph_evaluator_cli.py [options]
"""

import sys
import os
import argparse
from pathlib import Path

# Add the project root to Python path
project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from needle_evaluator_graph import NeedleEvaluatorGraph
from backend.agents.router_agent.router import create_router_from_config
from backend.core.api_utils import get_llm_langchain_openai


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate NeedleAgent Graph RAG performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python needle_graph_evaluator_cli.py
  python needle_graph_evaluator_cli.py --dataset custom_dataset.jsonl
  python needle_graph_evaluator_cli.py --output my_results.json
  python needle_graph_evaluator_cli.py --model gpt-4o --verbose
        """
    )
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="backend/data/evaluation_datasets/needle_agent/client3_report3_MenoraPolicy.jsonl",
        help="Path to evaluation dataset (default: %(default)s)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="needle_graph_evaluation_results.json",
        help="Output file for results (default: %(default)s)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model to use for evaluation (default: %(default)s)"
    )
    
    parser.add_argument(
        "--router-config",
        type=str,
        default="backend/agents/router_agent/config.yaml",
        help="Router configuration file (default: %(default)s)"
    )
    
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=["faithfulness", "context_recall", "llm_context_precision_with_reference"],
        help="Specific metrics to evaluate (default: all metrics)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )
    
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Only test setup, don't run full evaluation"
    )
    
    return parser.parse_args()


def test_setup(args):
    """Test the basic setup without running full evaluation."""
    print("🧪 Testing NeedleAgent Graph RAG Setup...")
    print("=" * 50)
    
    try:
        # Test 1: Dataset path
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            print(f"❌ Dataset not found at: {dataset_path}")
            return False
        print(f"✅ Dataset found: {dataset_path}")
        
        # Test 2: LLM initialization
        print("🔧 Testing LLM initialization...")
        llm = get_llm_langchain_openai(model=args.model)
        print(f"✅ LLM initialized: {args.model}")
        
        # Test 3: Router creation
        print("🔧 Testing router creation...")
        router = create_router_from_config(args.router_config)
        print("✅ Router created")
        
        # Test 4: Agent type detection
        agent_type = type(router.needle_agent).__name__
        print(f"🤖 Agent type: {agent_type}")
        
        if "Graph" not in agent_type:
            print("⚠️  Warning: Not using NeedleAgentGraph!")
            print("   Check your config.yaml - ensure NeedleAgent.type: graph and GraphRAG.enabled: true")
        else:
            print("🎯 Confirmed: Using NeedleAgentGraph (Graph RAG)")
        
        # Test 5: Evaluator creation
        print("🔧 Testing evaluator creation...")
        evaluator = NeedleEvaluatorGraph(dataset_path, llm, args.metrics)
        print("✅ Evaluator created successfully")
        
        # Test 6: Simple query test
        if router.needle_agent:
            print("🔧 Testing simple query...")
            test_response = router.needle_agent.answer("What is the test question?")
            print("✅ Agent query test passed")
            if args.verbose:
                print(f"   Response keys: {list(test_response.keys())}")
        
        print("\n🎉 All tests passed! Setup is ready for evaluation.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


def run_evaluation(args):
    """Run the full evaluation."""
    print("🚀 Starting NeedleAgent Graph RAG Evaluation...")
    print("=" * 50)
    
    try:
        # Check dataset
        dataset_path = Path(args.dataset)
        if not dataset_path.exists():
            print(f"❌ Dataset not found at: {dataset_path}")
            return False
        
        print(f"📊 Dataset: {dataset_path}")
        print(f"🎯 Output: {args.output}")
        print(f"🤖 Model: {args.model}")
        print(f"⚙️  Router Config: {args.router_config}")
        if args.metrics:
            print(f"📋 Metrics: {', '.join(args.metrics)}")
        print()
        
        # Initialize LLM
        print("🔧 Initializing LLM...")
        llm = get_llm_langchain_openai(model=args.model)
        print(f"✅ LLM initialized: {args.model}")
        
        # Create router
        print("🔧 Creating router...")
        router = create_router_from_config(args.router_config)
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
        evaluator = NeedleEvaluatorGraph(dataset_path, llm, args.metrics)
        print("✅ Evaluator created")
        
        # Run evaluation
        print("\n📈 Running evaluation...")
        if args.verbose:
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
            if args.verbose:
                print(f"🔍 DataFrame shape: {df_results.shape}")
                print(f"🔍 DataFrame columns: {list(df_results.columns)}")
            
            if not df_results.empty:
                metrics_row = df_results.iloc[0]
                if args.verbose:
                    print(f"🔍 First row: {dict(metrics_row)}")
                
                faithfulness = metrics_row.get('faithfulness', 'N/A')
                context_recall = metrics_row.get('context_recall', 'N/A')
                llm_context_precision = metrics_row.get('llm_context_precision_with_reference', 'N/A')
                
                print(f"Faithfulness: {faithfulness:.4f}" if isinstance(faithfulness, (int, float)) else f"Faithfulness: {faithfulness}")
                print(f"Context Recall: {context_recall:.4f}" if isinstance(context_recall, (int, float)) else f"Context Recall: {context_recall}")
                print(f"LLM Context Precision: {llm_context_precision:.4f}" if isinstance(llm_context_precision, (int, float)) else f"LLM Context Precision: {llm_context_precision}")
            else:
                print("No metrics available in results")
        else:
            print(f"Raw result: {result}")
        
        # Save results
        output_path = Path(args.output)
        print(f"\n💾 Saving results to {output_path}...")
        evaluator.save_results(result, output_path, router.needle_agent)
        print("✅ Results saved successfully!")
        
        print("\n🎉 Evaluation completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return False


def main():
    """Main CLI function."""
    args = parse_arguments()
    
    if args.test_only:
        success = test_setup(args)
    else:
        success = run_evaluation(args)
    
    if not success:
        print("\n❌ Operation failed. Check the error messages above.")
        if args.verbose:
            print("Use --verbose for more detailed error information.")
        sys.exit(1)
    else:
        print("\n✅ Operation completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()

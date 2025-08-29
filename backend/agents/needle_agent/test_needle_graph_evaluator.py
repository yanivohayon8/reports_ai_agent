#!/usr/bin/env python3
"""
Test script to verify NeedleEvaluatorGraph setup.
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


def test_setup():
    """Test the basic setup of NeedleEvaluatorGraph."""
    
    print("🧪 Testing NeedleEvaluatorGraph Setup...")
    
    try:
        # Test 1: LLM initialization
        print("1️⃣ Testing LLM initialization...")
        llm = get_llm_langchain_openai(model="gpt-4o-mini")
        print("✅ LLM initialized successfully")
        
        # Test 2: Router creation
        print("2️⃣ Testing router creation...")
        router = create_router_from_config()
        print("✅ Router created successfully")
        
        # Test 3: Agent type detection
        print("3️⃣ Testing agent type detection...")
        agent_type = type(router.needle_agent).__name__
        print(f"✅ Agent type: {agent_type}")
        
        if "Graph" not in agent_type:
            print("⚠️  Warning: Not using NeedleAgentGraph. Check your config.yaml")
            print("   Make sure NeedleAgent.type: graph and GraphRAG.enabled: true")
        
        # Test 4: Dataset path
        print("4️⃣ Testing dataset path...")
        dataset_path = Path("backend/data/evaluation_datasets/needle_agent/client3_report3_MenoraPolicy.jsonl")
        if dataset_path.exists():
            print("✅ Dataset found")
        else:
            print("❌ Dataset not found")
            return False
        
        # Test 5: Evaluator creation
        print("5️⃣ Testing evaluator creation...")
        evaluator = NeedleEvaluatorGraph(dataset_path, llm)
        print("✅ Evaluator created successfully")
        
        # Test 6: Simple query test
        print("6️⃣ Testing simple query...")
        if router.needle_agent:
            test_response = router.needle_agent.answer("What is the test question?")
            print("✅ Agent query test passed")
            print(f"   Response keys: {list(test_response.keys())}")
        else:
            print("❌ No needle agent available")
            return False
        
        print("\n🎉 All tests passed! NeedleEvaluatorGraph is ready to use.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_setup()
    sys.exit(0 if success else 1)

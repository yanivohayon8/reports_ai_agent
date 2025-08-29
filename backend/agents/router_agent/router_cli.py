import asyncio
import yaml
import sys
import os
from pathlib import Path

# Add the project root to Python path for direct execution
project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.core.user_interface import ConsoleChat
from backend.agents.router_agent.router import create_router_from_config


def build_router():
    config_path = "backend/agents/router_agent/config.yaml"
    
    return create_router_from_config(config_path)


def main():
    router = build_router()
    print("DEBUG: Agents availability → "
      f"Summary: {router.summary_agent is not None}, "
      f"Needle: {router.needle_agent is not None}, "
      f"Table: {router.table_agent is not None}, "
      f"Graph: {router.graph_indexer is not None}")
    
    def process_query(q: str) -> str:
        result = asyncio.run(router.handle(q))
        if isinstance(result, dict) and "answer" in result:
            return result["answer"]
        elif isinstance(result, str):
            return result
        else:
            return str(result)

    chat = ConsoleChat(process_query)
    chat.start()


if __name__ == "__main__":
    main()

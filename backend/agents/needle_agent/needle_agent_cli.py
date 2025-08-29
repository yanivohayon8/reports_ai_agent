import sys
sys.path.append("./")

from backend.core.user_interface import ConsoleChat
from backend.agents.needle_agent.needle_agent import NeedleAgent
from backend.indexer.indexer import FAISSIndexer
from backend.core.config_utils import load_config
from backend.core.api_utils import get_llm_langchain_openai

def main():
    config = load_config("backend/agents/needle_agent/needle_agent_config.yaml")
    faiss_config = config["faiss_indexer"]
    faiss_indexer = FAISSIndexer.from_small_embedding(directory_path=faiss_config["directory"])
    llm = get_llm_langchain_openai(model=config["llm"]["model"])
    needle_agent = NeedleAgent(faiss_indexer,llm)
    chat = ConsoleChat(needle_agent.answer)
    chat.start()


if __name__ == "__main__":
    main()
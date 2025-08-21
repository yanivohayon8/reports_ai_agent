import sys
sys.path.append("./")

from core.user_interface import ConsoleChat
from agents.needle_agent.needle import NeedleAgent
from indexer.indexer import FAISSIndexer
from core.config_utils import load_config
from core.api_utils import get_llm_langchain_openai

def main():
    config = load_config("agents/needle_agent/config.yaml")
    faiss_config = config["faiss_indexer"]
    faiss_indexer = FAISSIndexer.from_small_embedding(directory_path=faiss_config["directory"])
    llm = get_llm_langchain_openai(model=config["llm"]["model"])
    needle_agent = NeedleAgent(faiss_indexer,llm)
    chat = ConsoleChat(needle_agent.answer)
    chat.start()


if __name__ == "__main__":
    main()
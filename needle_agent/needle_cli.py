import sys
sys.path.append("./")

from core.user_interface import ConsoleChat
from needle_agent.needle import NeedleAgent
from indexer.indexer import FAISSIndexer
from core.config_utils import load_config

def main():
    config = load_config("needle_agent/config.yaml")
    faiss_config = config["faiss_indexer"]
    faiss_indexer = FAISSIndexer.from_small_embedding(directory_path=faiss_config["directory"])
    needle_agent = NeedleAgent(faiss_indexer)
    chat = ConsoleChat(needle_agent.search)
    chat.start()


if __name__ == "__main__":
    main()
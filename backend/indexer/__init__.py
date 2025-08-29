import yaml
from pathlib import Path

def init_indexer(config: dict, agent_name: str):
    idx_config = config[agent_name]["indexer"]

    if idx_config["type"] == "faiss":
        from backend.indexer.indexer import FAISSIndexer
        return FAISSIndexer.from_small_embedding(
            directory_path=idx_config["faiss_directory"]
        )
    elif idx_config["type"] == "graph":
        from backend.indexer.graph_indexer import GraphIndexer
        graph_indexer = GraphIndexer()
        # here we can load existing documents according to graph_nodes
        for node_name in idx_config.get("graph_nodes", []):
            # placeholder - in the future we will load real documents
            from llama_index.core import Document
            docs = [Document(f"Dummy document for {node_name}")]
            graph_indexer.add_documents(docs, node_name)
        graph_indexer.build_graph()
        return graph_indexer
    else:
        raise ValueError(f"Unknown indexer type {idx_config['type']}")

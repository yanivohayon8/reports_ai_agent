import networkx as nx
from typing import List, Any
from langchain_core.documents import Document

class NetworkXIndexer:
    def __init__(self, verbose: bool = True):
        self.graph = nx.Graph()
        self.verbose = verbose

    def add_documents(self, docs: List[Document], source_name: str):
        """
        Add chunks from a document as nodes, and link them by metadata.
        """
        for i, d in enumerate(docs):
            node_id = f"{source_name}_chunk_{i}"
            self.graph.add_node(node_id,
                                text=d.page_content,
                                **d.metadata)

            # --- Edges ---
            # קשר פנימי: אם אותו source
            if i > 0:
                prev_id = f"{source_name}_chunk_{i-1}"
                self.graph.add_edge(node_id, prev_id, type="same_doc")

            # קשר לפי IncidentType
            if "IncidentType" in d.metadata:
                self.graph.add_edge(node_id,
                                    d.metadata["IncidentType"],
                                    type="incident")

            # קשר לפי Policy
            if "PolicyName" in d.metadata:
                self.graph.add_edge(node_id,
                                    d.metadata["PolicyName"],
                                    type="policy")

            # קשר לפי TableId
            if "TableId" in d.metadata:
                self.graph.add_edge(node_id,
                                    f"table_{d.metadata['TableId']}",
                                    type="table")

        if self.verbose:
            print(f"[NetworkXIndexer] Added {len(docs)} nodes from {source_name}")

    def neighbors_text(self, node_id: str, depth: int = 1):
        """
        Return text of a node and its neighbors up to a given depth.
        """
        if node_id not in self.graph:
            return None

        sub_nodes = nx.single_source_shortest_path_length(self.graph, node_id, cutoff=depth).keys()
        return [self.graph.nodes[n].get("text", "") for n in sub_nodes if "text" in self.graph.nodes[n]]

    def visualize(self, out_path: str = "graph.png"):
        """
        Save a visualization of the graph.
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph, seed=42)
        nx.draw(self.graph, pos, with_labels=False, node_size=50, edge_color="gray")
        plt.savefig(out_path)
        plt.close()
        if self.verbose:
            print(f"[NetworkXIndexer] Graph visualization saved to {out_path}")

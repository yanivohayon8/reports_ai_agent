import os
import time
import json
from typing import List, Any
from .base_indexer import BaseIndexer
from llama_index.core import (
    Document as LlamaDocument,
    VectorStoreIndex,
    ComposableGraph,
    StorageContext,
    load_index_from_storage
)


class GraphIndexer(BaseIndexer):
    def __init__(self, verbose: bool = True):
        self.indices = {}
        self.graph = None
        self.metadata = {
            "type": "graph",
            "nodes": 0,
            "indices": []
        }
        self.verbose = verbose

    def add_documents(self, documents: List[LlamaDocument], index_name: str):
        if self.verbose:
            print(f"[GraphIndexer] Adding {len(documents)} documents to index '{index_name}'")
        index = VectorStoreIndex.from_documents(documents)
        self.indices[index_name] = index
        self.metadata["indices"].append(index_name)
        self.metadata["nodes"] += len(documents)

    def build_graph(self):
        if not self.indices:
            raise ValueError("No indices added to build graph.")

        if self.verbose:
            print(f"[GraphIndexer] Building graph with {len(self.indices)} indices...")
            print(f"[GraphIndexer] Indices: {list(self.indices.keys())}")

        start = time.time()
        self.graph = ComposableGraph.from_indices(
            root_index_cls=VectorStoreIndex,
            children_indices=list(self.indices.values()),
            index_summaries=[f"Index for {name}" for name in self.indices.keys()],
        )
        elapsed = time.time() - start

        if self.verbose:
            print(f"[GraphIndexer] Graph build complete ✓")
            print(f"[GraphIndexer] Total time: {elapsed:.2f} seconds")

    def retrieve(self, query: str, **kwargs) -> dict:
        """
        Run a query on the graph and return both the answer and sources (if available).
        """
        if not self.graph:
            raise RuntimeError("Graph not built. Call build_graph() first.")

        if self.verbose:
            print(f"[GraphIndexer] Retrieving answer for query: '{query}'")

        # Use a slightly higher candidate pool to avoid missing named entities
        query_engine = self.graph.as_query_engine(similarity_top_k=8)
        response = query_engine.query(query)

        # Always return answer
        result = {"answer": str(response), "sources": []}

        # Try to extract sources if available
        if hasattr(response, "source_nodes") and response.source_nodes:
            for n in response.source_nodes:
                try:
                    meta = getattr(n.node, "metadata", {})
                    src = meta.get("source", "unknown")
                    page = meta.get("page", "n/a")
                    snippet = n.node.get_content()[:150]
                    result["sources"].append(
                        {"source": src, "page": page, "snippet": snippet}
                    )
                except Exception as e:
                    if self.verbose:
                        print("[GraphIndexer] Error extracting source:", e)

        return result

    def save(self, directory_path: str):
        """
        Save all child indices and metadata to disk (not the graph object directly).
        """
        if not self.indices:
            raise RuntimeError("No indices to save.")
        os.makedirs(directory_path, exist_ok=True)

        index_map = {}
        for name, index in self.indices.items():
            idx_path = os.path.join(directory_path, name.replace(":", "_"))
            if self.verbose:
                print(f"[GraphIndexer] Saving index '{name}' to {idx_path}")
            index.storage_context.persist(idx_path)
            index_map[name] = idx_path

        # save metadata
        meta_path = os.path.join(directory_path, "graph_metadata.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump({"indices": index_map, "meta": self.metadata}, f, indent=2)

        if self.verbose:
            print(f"[GraphIndexer] Graph indices saved to {directory_path}")

    @classmethod
    def load(cls, directory_path: str, verbose: bool = True):
        """
        Load graph from saved indices (ComposableGraph has no load_from_disk).
        """
        meta_path = os.path.join(directory_path, "graph_metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"No graph_metadata.json found in {directory_path}")

        with open(meta_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        gi = cls(verbose=verbose)
        index_objs = {}

        for name, idx_path in data["indices"].items():
            if verbose:
                print(f"[GraphIndexer] Loading index '{name}' from {idx_path}")
            storage_context = StorageContext.from_defaults(persist_dir=idx_path)
            index = load_index_from_storage(storage_context)
            index_objs[name] = index

        gi.indices = index_objs
        gi.metadata = data.get("meta", {})

        gi.graph = ComposableGraph.from_indices(
            root_index_cls=VectorStoreIndex,
            children_indices=list(index_objs.values()),
            index_summaries=[f"Index for {name}" for name in index_objs.keys()]
        )
        return gi

    def get_used_input(self) -> dict:
        return self.metadata

    # --- Helpers to ingest from existing LangChain docs ---
    @staticmethod
    def _lc_to_llama_docs(langchain_docs: List[Any]) -> List[LlamaDocument]:
        llama_docs: List[LlamaDocument] = []
        for d in langchain_docs:
            try:
                text = getattr(d, "page_content", None) or getattr(d, "content", "")
                meta = getattr(d, "metadata", {}) or {}

                # ✅ Clean metadata
                safe_meta = {}
                allowed_keys = ["source", "page", "TableId", "FigureId", "IncidentType", "IncidentDate"]
                for k in allowed_keys:
                    if k in meta:
                        v = meta[k]
                        if isinstance(v, (str, int, float, bool)):
                            safe_meta[k] = v
                        else:
                            safe_meta[k] = str(v)[:200]

                if "doc_id" in meta:
                    safe_meta["doc_id"] = str(meta["doc_id"])

                llama_docs.append(LlamaDocument(text=text, metadata=safe_meta))

            except Exception as e:
                print(f"[GraphIndexer] Error converting document: {e}")
                continue
        return llama_docs

    def add_langchain_documents(self, langchain_docs: List[Any], index_name: str):
        llama_docs = self._lc_to_llama_docs(langchain_docs)
        if not llama_docs:
            if self.verbose:
                print(f"[GraphIndexer] No documents converted for index '{index_name}'")
            return
        self.add_documents(llama_docs, index_name)

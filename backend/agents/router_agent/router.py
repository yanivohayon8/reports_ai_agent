import yaml
import sys
import os
from pathlib import Path

# Add the project root to Python path for direct execution
project_root = os.path.join(os.path.dirname(__file__), '..', '..', '..')
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.indexer.indexer import FAISSIndexer
from backend.agents.summary_agent.summary_agent import SummaryAgent
from backend.agents.summary_agent.summary_chunker import SummaryChunker
from backend.agents.needle_agent.needle_agent import NeedleAgent
from backend.agents.tableQA_agent.tableQA_agent import TableQAgent
from backend.core.api_utils import get_llm_langchain_openai
from backend.retrieval.hybrid_retriever import HybridRetriever
from backend.retrieval.dense_retriever import DenseRetriever
from backend.retrieval.sparse_retriever import SparseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Optional Graph RAG support
try:
    from backend.indexer.graph_indexer import GraphIndexer  # type: ignore
except Exception:
    GraphIndexer = None  # type: ignore


class RouterAgent:
    def __init__(self, model_name="gpt-4o-mini",
                 summary_agent: SummaryAgent = None,
                 needle_agent: NeedleAgent = None,
                 table_agent: TableQAgent = None,
                 graph_indexer=None):
        self.model_name = model_name
        self.summary_agent = summary_agent
        self.needle_agent = needle_agent
        self.table_agent = table_agent
        self.graph_indexer = graph_indexer
        self.llm = get_llm_langchain_openai(model=model_name)

    def is_table_question(self, query: str) -> bool:
        """
        Detect if the query is about table data.
        Uses hybrid retrieval to check if top docs are table chunks.
        """
        if not self.table_agent or not hasattr(self.table_agent, "retriever"):
            return False

        try:
            # Increase retrieval to get better coverage
            docs = self.table_agent.retriever.retrieve(query, k_dense=5, k_sparse=5)
        except Exception as e:
            print(f"Router: is_table_question retrieval error: {e}")
            return False

        if not docs:
            return False

        # Check for table chunks
        table_hits = sum(
            1 for d in docs if d.metadata.get("table_chunk_type") in ["raw", "description"]
        )

        # Also check if any document content looks like table data
        table_content_hits = sum(
            1 for d in docs if "|" in d.page_content and ("---" in d.page_content or any(char.isdigit() for char in d.page_content[:200]))
        )

        total_table_evidence = max(table_hits, table_content_hits)
        
        # More lenient threshold: if we have any table evidence, consider it a table question
        return total_table_evidence > 0

    async def handle(self, query: str):
        """
        Classification logic with retrieval-aware overrides and fallback.
        """
        q = query.lower()

        # Indicators
        needle_indicators = [
            "find", "search", "locate", "when", "where", "how much",
            "date", "time", "location", "amount", "cost", "price",
            "policy number", "hospital", "surgery",
            "specific", "exact", "precise", "details about", "information on",
            "occur", "athens", "city", "place", "what is", "who is", "which",
            "nationality", "age", "birth", "country", "origin", "from"
        ]
        table_indicators = [
            "table", "chart", "data", "statistics", "numbers",
            "figures", "columns", "rows", "cells", "spreadsheet", "grid",
            "compare", "calculate", "total", "sum", "average", "percentage",
            "coverage", "limit", "liability", "deductible", "policy", "claim",
            "incident", "accident", "report", "details", "information"
        ]
        summary_indicators = [
            "summarize", "overview", "context", "in short", "explain", "summary",
            "tell me about", "describe", "background", "story"
        ]

        # Scores
        needle_score = sum(1 for ind in needle_indicators if ind in q)
        table_score = sum(1 for ind in table_indicators if ind in q)
        summary_score = sum(1 for ind in summary_indicators if ind in q)

        # Boost needle score for specific question patterns
        if any(pattern in q for pattern in ["what is", "who is", "which", "where", "when", "how much"]):
            needle_score += 2

        # Boost table score for data-specific questions
        if any(pattern in q for pattern in ["nationality", "age", "birth", "country", "policy", "claim", "accident"]):
            table_score += 1

        # Initial classification
        if table_score > needle_score:
            rtype = "table"
        elif needle_score > 0:
            rtype = "needle"
        elif summary_score > 0:
            rtype = "summary"
        else:
            # Default to needle for specific fact questions, summary for general ones
            if any(pattern in q for pattern in ["what", "who", "which", "where", "when", "how"]):
                rtype = "needle"
            else:
                rtype = "summary"

        # Debug
        print(f"Router: Query '{query}' initial classification → {rtype}")
        print(f"Router: Needle score={needle_score}, Table score={table_score}, Summary score={summary_score}")
        print(f"Router: Agents - Summary: {self.summary_agent is not None}, "
              f"Needle: {self.needle_agent is not None}, Table: {self.table_agent is not None}")

        # Check if answer involves table data before final classification
        # Always check for table data when query mentions tables, regardless of initial classification
        if any(word in q for word in table_indicators):
            if self.is_table_question(query):
                print("Router override: → Table (retriever evidence shows table data)")
                rtype = "table"
            else:
                print("Router: Query mentions tables but no table data found, keeping as:", rtype)

        # Dispatch
        if rtype == "summary" and self.summary_agent:
            return await self.summary_agent.handle(query)
        elif rtype == "needle" and self.needle_agent:
            return await self.needle_agent.handle(query)
        elif rtype == "table" and self.table_agent:
            answer = await self.table_agent.handle(query)
            # Fallback if no relevant table was found
            if isinstance(answer, dict) and "answer" in answer and "no relevant table" in answer["answer"].lower():
                print("Router fallback: table returned nothing → retrying as summary")
                if self.summary_agent:
                    return await self.summary_agent.handle(query)
            return answer
        else:
            return {"answer": "No suitable agent available.", "agent": "Router"}

    def graph_retrieve(self, query: str) -> dict:
        """
        Direct Graph RAG retrieval (bypasses routing).
        """
        if not self.graph_indexer:
            return {"answer": "Graph RAG not configured.", "agent": "Router"}
        try:
            result = self.graph_indexer.retrieve(query)
            return {"answer": str(result), "agent": "GraphRAG"}
        except Exception as e:
            return {"answer": f"Graph query failed: {e}", "agent": "GraphRAG"}


def create_router_from_config(config_path: str = "backend/agents/router_agent/config.yaml") -> RouterAgent:
    """
    Build RouterAgent and its sub-agents directly from config.yaml
    """
    try:
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Error loading config: {e}")
        return RouterAgent()  # minimal router

    summary_agent = None
    needle_agent = None
    table_agent = None
    graph_indexer = None

    try:
        print("Router: Starting agent initialization...")

        # LLMs
        print("Router: Initializing LLMs...")
        llm_summary = get_llm_langchain_openai(model=config["SummaryAgent"]["llm"]["model"])
        llm_needle = get_llm_langchain_openai(model=config["NeedleAgent"]["llm"]["model"])
        llm_table = get_llm_langchain_openai(model=config["TableQAgent"]["llm"]["model"])
        print("Router: LLMs initialized successfully")

        # Table FAISS Indexer
        print("Router: Initializing Table indexer...")
        table_indexer = FAISSIndexer.from_small_embedding(
            directory_path=config["TableQAgent"]["indexer"]["faiss_directory"]
        )
        print(f"Router: Table indexer initialized from {config['TableQAgent']['indexer']['faiss_directory']}")

        # Optional Graph RAG
        try:
            if config.get("GraphRAG", {}).get("enabled", False) and GraphIndexer is not None:
                graph_indexer = GraphIndexer.load(config["GraphRAG"]["directory"])
                print(f"Router: Graph RAG loaded ✓ from {config['GraphRAG']['directory']}")
        except Exception as e:
            print(f"Router: Graph RAG setup error: {e}")

        # Create HybridRetriever for TableQAgent
        print("Router: Creating hybrid retriever...")
        dense_retriever = DenseRetriever(table_indexer.vector_store)
        try:
            dummy_docs = table_indexer.retrieve("dummy", num_documents=20)
            if dummy_docs and len(dummy_docs) > 0:
                sparse_retriever = SparseRetriever(dummy_docs)
                hybrid_retriever = HybridRetriever(dense_retriever, sparse_retriever)
                print("Router: Hybrid retriever created successfully with sparse retriever")
            else:
                print("Router: No documents found, using dense retriever only")
                hybrid_retriever = dense_retriever
        except Exception as e:
            print(f"Router: Error creating sparse retriever: {e}")
            print("Router: Using dense retriever only as fallback")
            hybrid_retriever = dense_retriever

        # === Agents ===
        print("Router: Creating agents...")

        # SummaryAgent
        if config["SummaryAgent"].get("type", "faiss") == "graph" and graph_indexer:
            from backend.agents.summary_agent.summary_agent_graph import SummaryAgentGraph as SummaryAgentImpl
            print("Router: Using SummaryAgentGraph (GraphIndexer)")
            summary_agent = SummaryAgentImpl(graph_indexer, llm_summary)
        else:
            from backend.agents.summary_agent.summary_agent import SummaryAgent as SummaryAgentImpl
            print("Router: Using SummaryAgent (FAISSIndexer + Chunker)")
            summary_indexer = FAISSIndexer.from_small_embedding(
                directory_path=config["SummaryAgent"]["indexer"]["faiss_directory"]
            )
            chunker_conf = config["SummaryAgent"]["chunker"]
            summary_chunker = SummaryChunker(
                faiss_indexer=summary_indexer,
                text_splitter=RecursiveCharacterTextSplitter(
                    chunk_size=chunker_conf["chunk_size"],
                    chunk_overlap=chunker_conf["chunk_overlap"],
                    separators=chunker_conf["separators"]
                )
            )
            summary_agent = SummaryAgentImpl(chunker=summary_chunker, llm=llm_summary)

        # NeedleAgent
        if config["NeedleAgent"].get("type", "faiss") == "graph" and graph_indexer:
            from backend.agents.needle_agent.needle_agent_graph import NeedleAgentGraph as NeedleAgentImpl
            print("Router: Using NeedleAgentGraph (GraphIndexer)")
            needle_agent = NeedleAgentImpl(graph_indexer, llm_needle)
        else:
            from backend.agents.needle_agent.needle_agent import NeedleAgent as NeedleAgentImpl
            print("Router: Using NeedleAgent (FAISSIndexer)")
            needle_indexer = FAISSIndexer.from_small_embedding(
                directory_path=config["NeedleAgent"]["indexer"]["faiss_directory"]
            )
            needle_agent = NeedleAgentImpl(needle_indexer, llm_needle)

        # TableAgent (always FAISS+HybridRetriever)
        from backend.agents.tableQA_agent.tableQA_agent import TableQAgent
        table_agent = TableQAgent(hybrid_retriever, llm_table)

        print("Router: All agents created successfully")
        print(f"Router: Successfully initialized agents - "
              f"Summary: ✓, Needle: ✓, Table: ✓, Graph: {'✓' if graph_indexer else '✗'}")

    except Exception as e:
        print(f"Error initializing agents: {e}")
        import traceback
        traceback.print_exc()
        print("Router: Some agents may not be available")

    return RouterAgent(
        model_name=config.get("Router", {}).get("model_name", "gpt-4o-mini"),
        summary_agent=summary_agent,
        needle_agent=needle_agent,
        table_agent=table_agent,
        graph_indexer=graph_indexer
    )

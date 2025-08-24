from fastapi import FastAPI
from pydantic import BaseModel
import asyncio

import sys
import os

# Add the backend directory to Python path
backend_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, backend_path)

from agents.router_agent.router import RouterAgent
from retrieval import DenseRetriever, SparseRetriever, HybridRetriever
from indexer.indexer import FAISSIndexer
from core.config_utils import load_config

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Reports AI Agent API")

# ✅ Enable CORS so frontend can talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Allow both ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"msg": "Backend is running 🚀"}

# ✅ Load config & initialize retrievers
config = load_config("agents/router_agent/config.yaml")
faiss_indexer = FAISSIndexer.from_small_embedding(
    directory_path=config["faiss_indexer"]["directory"]
)
dense = DenseRetriever(faiss_indexer.vector_store)
docs = faiss_indexer.retrieve("dummy", num_documents=20)
sparse = SparseRetriever(docs)
hybrid = HybridRetriever(dense, sparse)
router_agent = RouterAgent(retriever=hybrid, faiss_indexer=faiss_indexer, model_name=config["llm"]["model"])

@app.post("/chat")
async def chat(query: Query):
    try:
        print(f"Chat endpoint: Received query: {query.query}")
        result = await router_agent.handle(query.query)
        print(f"Chat endpoint: Router result: {result}")
        
        # Handle both string and dictionary responses
        if isinstance(result, dict):
            response = {
                "answer": result.get("answer", str(result)),
                "agent": result.get("agent", "🤖"),
                "reasoning": result.get("reasoning", "AI processed your request")
            }
            print(f"Chat endpoint: Returning response: {response}")
            return response
        else:
            # If result is a string or other type, convert it
            response = {
                "answer": str(result),
                "agent": "🤖",
                "reasoning": "AI processed your request"
            }
            print(f"Chat endpoint: Returning response: {response}")
            return response
    except Exception as e:
        print(f"Chat endpoint: Error occurred: {str(e)}")
        return {
            "answer": f"Error: {str(e)}",
            "agent": "❌",
            "reasoning": "Backend error occurred"
        }

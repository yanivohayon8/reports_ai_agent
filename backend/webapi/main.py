from fastapi import FastAPI
from pydantic import BaseModel
import os
import sys

# Ensure the project root (parent of the `backend` package) is on sys.path
current_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.agents.router_agent.router import create_router_from_config
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Reports AI Agent API")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str

@app.get("/")
def read_root():
    return {"msg": "Backend is running 🚀"}

def build_router():
    # Mirror router_cli.py behavior
    project_root = os.path.join(os.path.dirname(__file__), "..", "..")
    try:
        os.chdir(project_root)
    except Exception:
        pass
    config_path = "backend/agents/router_agent/config.yaml"
    return create_router_from_config(config_path)


print("Web API: Initializing router agent...")
try:
    router_agent = build_router()
    print("Web API: Router agent initialized successfully")
except Exception as e:
    print(f"Web API: Failed to initialize router agent: {e}")
    router_agent = None

@app.post("/chat")
async def chat(query: Query):
    if not router_agent:
        return {
            "answer": "Router agent not available. Please check backend logs.",
            "agent": "❌",
            "reasoning": "Router initialization failed",
            "response_type": "error"
        }

    try:
        result = await router_agent.handle(query.query)

        # Default minimal response (CLI-like)
        if not isinstance(result, dict):
            return {"answer": str(result), "response_type": "text"}

        agent_name = result.get("agent", "Unknown")
        answer = result.get("answer", str(result))
        reasoning = result.get("reasoning", "AI processed your request")

        # Enhanced table visualization when TableQAgent is used
        if agent_name == "TableQAgent":
            table_data = extract_table_data(result)
            if table_data and not table_data.get("error"):
                return {
                    "answer": answer,
                    "agent": agent_name,
                    "reasoning": reasoning,
                    "table_data": table_data,
                    "response_type": "table"
                }

        # Fallback to plain text response
        return {
            "answer": answer,
            "agent": agent_name,
            "reasoning": reasoning,
            "response_type": "text"
        }
    except Exception as e:
        print(f"Chat endpoint: Error occurred: {str(e)}")
        return {
            "answer": f"Error: {str(e)}",
            "agent": "❌",
            "reasoning": "Backend error occurred",
            "response_type": "error"
        }


@app.post("/graph")
async def graph(query: Query):
    if not router_agent:
        return {"answer": "Router not available."}
    try:
        result = router_agent.graph_retrieve(query.query)
        if isinstance(result, dict) and "answer" in result:
            return {"answer": result["answer"], "agent": result.get("agent", "GraphRAG")}
        return {"answer": str(result)}
    except Exception as e:
        print(f"Graph endpoint error: {e}")
        return {"answer": f"Error: {str(e)}"}

def extract_table_data(result: dict) -> dict:
    """
    Extract table data from TableQAgent response.
    First tries to parse markdown table from answer text, then falls back to chunks.
    """
    try:
        # Method 1: Parse markdown table from answer text
        answer_text = result.get("answer", "")
        if answer_text and "|" in answer_text and "---" in answer_text:
            table_data = parse_markdown_table(answer_text)
            if table_data:
                return {
                    "raw_content": answer_text,
                    "parsed_tables": [table_data]
                }

        # Method 2: Try to get table from chunks
        chunks_content = result.get("chunks_content", [])
        if chunks_content:
            for content in chunks_content:
                if isinstance(content, str) and "|" in content and "---" in content:
                    table_data = parse_markdown_table(content)
                    if table_data:
                        return {
                            "raw_content": content,
                            "parsed_tables": [table_data]
                        }

        return {"error": "No table data found"}

    except Exception as e:
        print(f"Error extracting table data: {e}")
        return {"error": f"Failed to parse table data: {str(e)}"}


def parse_markdown_table(text: str) -> dict:
    """
    Parse a markdown table string into structured data.
    """
    try:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if len(lines) < 3:
            return None

        # Find the table structure
        table_start = None
        separator_line = None

        for i, line in enumerate(lines):
            if '|' in line:
                if table_start is None:
                    table_start = i
                elif separator_line is None and '-' in line and ':' in line:
                    separator_line = i
                    break

        if table_start is None or separator_line is None:
            return None

        # Parse headers
        header_line = lines[table_start]
        headers = [h.strip() for h in header_line.split('|')[1:-1]]

        if not headers or len(headers) < 2:
            return None

        # Parse data rows
        rows = []
        for line in lines[separator_line + 1:]:
            if '|' in line:
                values = [v.strip() for v in line.split('|')[1:-1]]
                if len(values) == len(headers):
                    row_dict = dict(zip(headers, values))
                    rows.append(row_dict)

        if not rows:
            return None

        return {
            "headers": headers,
            "rows": rows,
            "type": "markdown"
        }

    except Exception as e:
        print(f"Error parsing markdown table: {e}")
        return None

# Note: Web API now mirrors Router CLI by returning only an answer string.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

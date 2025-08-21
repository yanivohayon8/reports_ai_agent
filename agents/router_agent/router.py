from openai import OpenAI
from pydantic import BaseModel
from typing import Literal


client = OpenAI()

class Classification(BaseModel):
    reasoning: str
    type: Literal["tableQA", "summary", "needle"]
    complexity: Literal["simple", "complex"]
    
def classify_query(query: str) -> Classification:
    """Classify a query into tableQA, summary, or needle."""
    response = client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that classifies queries into "
                    "tableQA, summary, or needle. Respond ONLY with a JSON object "
                    "that matches the required schema."
                ),
            },
            {
                "role": "user",
                "content": f"""
    Classify this query:
    {query}

    Determine and output fields:
    - type: one of [summary, needle, tableQA]
    - complexity: one of [simple, complex]
    - reasoning: brief justification
    """,
            },
        ],
        response_format=Classification,
    )
    return response.choices[0].message.parsed

def generate_response(query: str, classification: Classification, context: str) -> str:
    """Generate a response to a query based on the classification and context."""
    system_prompts = {
        "summary": "You are an expert summarizer of accident/burglary insurance reports.",
        "needle": "You locate specific sections or clauses in insurance documents and return exact references.",
        "tableQA": "You analyze tabular/numerical data and return answers with table IDs and row anchors.",
    }
    model = "gpt-4o-mini" if classification.complexity == "simple" else "gpt-4o"
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompts[classification.type]},
            {"role": "user", "content": f"""
    Generate a response to this query:
    {query}
    
    Classification:
    {classification}
    
    Context:
    {context}
    """},
        ],
    )
    return response.choices[0].message.content      





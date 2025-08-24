import sys
import os
from dataclasses import dataclass
from typing import Literal, Optional, Any

# Add the backend directory to Python path
backend_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, backend_path)

from agents.summary_agent.summary import SummaryAgent
from agents.needle_agent.needle import NeedleAgent
from agents.tableQA_agent.tableQA import TableQAgent
from retrieval.hybrid_retriever import HybridRetriever
from langchain_core.prompts import ChatPromptTemplate
from core.api_utils import get_llm_langchain_openai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from indexer.indexer import FAISSIndexer

@dataclass
class Classification:
    reasoning: str
    type: Literal["summary", "needle", "tableQA"]
    complexity: Literal["simple", "complex"]


def classify_query(query: str) -> Classification:
    q = query.lower()
    if "table" in q:
        ctype = "tableQA"
        reason = "Query mentions table-related analysis."
    elif any(k in q for k in ["find", "date", "who", "what", "where"]):
        ctype = "needle"
        reason = "Query looks like a direct fact lookup."
    else:
        ctype = "summary"
        reason = "Defaulting to summarization for general queries."
    return Classification(reasoning=reason, type=ctype, complexity="simple")


def generate_response(query: str, classification: Classification, context: str) -> str:
    if classification.type == "summary":
        return f"Summary: {context}"
    if classification.type == "needle":
        return f"Answer: Based on context, {context}"
    if classification.type == "tableQA":
        return "Table Answer: Derived from table context."
    return "I could not generate a response."


class RouterAgent:
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        retriever: HybridRetriever = None,
        faiss_indexer: FAISSIndexer = None,
    ):
        # Simple rule-based routing; initialize only what's needed
        try:
            llm = get_llm_langchain_openai(model=model_name)
        except Exception:
            llm = None

        # Initialize agents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200) if llm is not None else None
        self.summary_agent = SummaryAgent(text_splitter=text_splitter, llm=llm) if llm is not None else None
        # Needle agent requires an indexer; only construct if both are available
        self.needle_agent = NeedleAgent(faiss_indexer, llm) if (faiss_indexer is not None and llm is not None) else None
        self.table_agent = TableQAgent(retriever=retriever)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a router that classifies user questions and decides which specialized agent should answer."),
            ("human",
             "Classify this user question:\n"
             "{query}\n\n"
             "Decide:\n"
             "1. reasoning (why you decided)\n"
             "2. type (summary / needle / table)")
        ])
        self.classify_template = prompt   
        self.llm = get_llm_langchain_openai()
        
    async def classify_via_llm(self, query: str):
        chain = self.classify_template | self.llm.with_structured_output(Classification)
        return await chain.ainvoke({"query": query})
    
    async def classify(self, query: str):
        c = classify_query(query)
        # Normalize to legacy "table" for internal routing while keeping public API as tableQA
        rtype = "table" if c.type == "tableQA" else c.type
        return type("RouterClassification", (), {"reasoning": c.reasoning, "type": rtype})()

    async def handle(self, query: str) -> str:
        # Use simple rule-based classification to avoid LLM dependency
        classification = await self.classify(query)
        rtype = classification.type

        if rtype == "summary":
            if self.summary_agent is not None:
                result = await self.summary_agent.handle(query)
                return {
                    "answer": result.get("answer", "No answer generated"),
                    "agent": result.get("agent", f"Router → {rtype.upper()}"),
                    "reasoning": result.get("reasoning", classification.reasoning),
                    "type": rtype
                }
            else:
                # Fallback non-LLM response for summary
                answer = generate_response(query, Classification("fallback", "summary", "simple"), "")
                return {
                    "answer": answer,
                    "agent": f"Router → {rtype.upper()}",
                    "reasoning": classification.reasoning,
                    "type": rtype
                }
        elif rtype == "needle":
            if self.needle_agent is not None:
                try:
                    result = await self.needle_agent.handle(query)
                    return {
                        "answer": result.get("answer", "No answer generated"),
                        "agent": result.get("agent", f"Router → {rtype.upper()}"),
                        "reasoning": result.get("reasoning", classification.reasoning),
                        "type": rtype
                    }
                except Exception as e:
                    return {
                        "answer": f"Error in needle agent: {str(e)}",
                        "agent": f"Router → {rtype.upper()}",
                        "reasoning": f"Agent error: {str(e)}",
                        "type": rtype
                    }
            else:
                answer = generate_response(query, Classification("fallback", "needle", "simple"), "")
                return {
                    "answer": answer,
                    "agent": f"Router → {rtype.upper()}",
                    "reasoning": classification.reasoning,
                    "type": rtype
                }
        elif rtype == "table":
            if self.table_agent is not None:
                result = await self.table_agent.handle(query)
                return {
                    "answer": result.get("answer", "No answer generated"),
                    "agent": result.get("agent", f"Router → {rtype.upper()}"),
                    "reasoning": result.get("reasoning", classification.reasoning),
                    "type": rtype
                }
            else:
                answer = "No table agent available."
                return {
                    "answer": answer,
                    "agent": f"Router → {rtype.upper()}",
                    "reasoning": classification.reasoning,
                    "type": rtype
                }
        else:
            answer = "I could not classify the question."
            return {
                "answer": answer,
                "agent": f"Router → {rtype.upper()}",
                "reasoning": classification.reasoning,
                "type": rtype
            }

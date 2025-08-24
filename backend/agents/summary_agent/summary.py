import pathlib
import sys
import os

# Add the backend directory to Python path
backend_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, backend_path)

from core.pdf_reader import read_pdf            
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .summary_prompts import MAP_SUMMARY_PROMPT_CHAT_TEMPLATE,REDUCE_SUMMARY_PROMPT_CHAT_TEMPLATE
from .summary_prompts import ITERATIVE_REFINEMENT_PROMPT_CHAT_TEMPLATE,ITERATIVE_REFINEMENT_INITIAL_SUMMARY_PROMPT_CHAT_TEMPLATE
from langchain_core.language_models.chat_models import BaseChatModel

class SummaryAgent:

    def __init__(self,text_splitter:RecursiveCharacterTextSplitter,llm:BaseChatModel):
        self.llm = llm
        self.text_splitter = text_splitter
        
    def _create_default_splitter(self):
        """Create a default text splitter if none provided"""
        if self.text_splitter is None:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            return RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return self.text_splitter
        
    async def handle(self, query: str) -> dict:
        """
        Adapter for RouterAgent.
        Returns structured response with answer and metadata.
        """
        result = self.summarize(query)
        return {
            "answer": result["answer"],
            "agent": "Summary Agent",
            "reasoning": "Generated summary from document content"
        }

    def summarize_single_pdf(self,pdf_path:pathlib.Path, method:str):
        text = read_pdf(pdf_path,format="text")
        return self.summarize(text,method)

    def summarize(self,text:str,method:str="map_reduce"):
        splitter = self._create_default_splitter()
        if method == "map_reduce":
            return self._summarize_map_reduce(text, splitter)
        elif method == "iterative":
            return self._summarize_iterative_refinement(text, splitter)
        else:
            raise ValueError(f"Invalid summary method: {method}")

    def _summarize_map_reduce(self,text:str, splitter):
        chunks = splitter.split_text(text)
        partial_summaries = self._summarize_map(chunks)
        summary = self._summarize_reduce(partial_summaries)
        return {"answer": summary}

    def _summarize_map(self,chunks:list[str]):
        map_chain = MAP_SUMMARY_PROMPT_CHAT_TEMPLATE | self.llm

        partial_summaries = []

        for chunk in chunks:
            response = map_chain.invoke({"text":chunk})
            partial_summaries.append(response.content)

        return partial_summaries
    
    def _summarize_reduce(self,partial_summaries:list[str]):
        reduce_chain = REDUCE_SUMMARY_PROMPT_CHAT_TEMPLATE | self.llm
        combined_text = "\n".join(partial_summaries)
        response = reduce_chain.invoke({"text":combined_text})
        return response.content
    
    def _summarize_iterative_refinement(self,text:str, splitter):
        chunks = splitter.split_text(text)
        initial_summary_chain = ITERATIVE_REFINEMENT_INITIAL_SUMMARY_PROMPT_CHAT_TEMPLATE | self.llm
        initial_summary = initial_summary_chain.invoke({"text":chunks[0]})
        refinement_chain = ITERATIVE_REFINEMENT_PROMPT_CHAT_TEMPLATE | self.llm

        for chunk in chunks[1:]:
            response = refinement_chain.invoke({"summary":initial_summary,"text":chunk})
            initial_summary = response.content
        
        return {"answer": initial_summary}





import sys
import os

# Add the backend directory to Python path
backend_path = os.path.join(os.path.dirname(__file__), '..', '..')
sys.path.insert(0, backend_path)

from indexer.indexer import FAISSIndexer
from typing import List
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.documents import Document
from .needle_prompts import generation_prompt_template


class NeedleAgent():
    
    def __init__(self, faiss_indexer:FAISSIndexer, llm:BaseChatModel) -> None:
        self.faiss_indexer = faiss_indexer
        self.llm = llm
        
    async def handle(self, query: str) -> dict:
        """
        Adapter for RouterAgent.
        Returns structured response with answer and metadata.
        """
        try:
            result = self.answer(query)
            return {
                "answer": result["answer"],
                "chunks": result.get("chunks", []),
                "agent": "Needle Agent",
                "reasoning": "Direct fact lookup from documents"
            }
        except Exception as e:
            return {
                "answer": f"Error processing query: {str(e)}",
                "chunks": [],
                "agent": "Needle Agent",
                "reasoning": f"Error occurred: {str(e)}"
            }


    def answer(self, query:str)->dict:
        try:
            print(f"NeedleAgent: Processing query: {query}")
            context,chunks = self._retrieve_context(query)
            print(f"NeedleAgent: Retrieved {len(chunks)} chunks, context length: {len(context)}")
            
            answer = self._generate(context, query)
            print(f"NeedleAgent: Generated answer: {answer[:100]}...")

            chunks_debug_info = self._get_chunks_debug_info(chunks)

            return {
                "answer":answer,
                "chunks":chunks_debug_info
            }
        except Exception as e:
            print(f"NeedleAgent: Error in answer method: {str(e)}")
            raise e

    def _retrieve_context(self, query:str)->tuple[str,List[Document]]:
        chunks = self.faiss_indexer.retrieve(query)
        context = self._concat_chunks(chunks)

        return context,chunks
        
    def _concat_chunks(self,chunks:List[Document])->str:
        return "\n\n".join([chunk.page_content for chunk in chunks])
    
    def _generate(self,context:str, query:str)->str:
        prompt = generation_prompt_template.invoke({"context":context,"query":query})
        answer = self.llm.invoke(prompt)

        return answer.content

    def _get_chunks_debug_info(self,chunks:List[Document])->str:
        return [{"page_content":chunk.page_content,"metadata":chunk.metadata} for chunk in chunks]

from indexer.indexer import FAISSIndexer
from typing import List
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.documents import Document
from agents.needle_agent.needle_prompts import generation_prompt_template


class NeedleAgent():
    
    def __init__(self, faiss_indexer:FAISSIndexer, llm:BaseChatModel) -> None:
        self.faiss_indexer = faiss_indexer
        self.llm = llm
        
    async def handle(self, query: str) -> str:
        """
        Adapter for RouterAgent.
        Returns only the answer (without debug info).
        """
        result = self.answer(query)
        return result["answer"]


    def answer(self, query:str)->dict:
        context,chunks = self._retrieve_context(query)
        answer = self._generate(context, query)

        chunks_content,chunks_metadata = self._get_chunks_info(chunks)

        return {
            "answer":answer,
            "chunks_content":chunks_content,
            "chunks_metadata":chunks_metadata
        }

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

    def _get_chunks_info(self,chunks:List[Document])->tuple[List[str],List[dict]]:
        chunks_content = [chunk.page_content for chunk in chunks]
        chunks_metadata = [chunk.metadata for chunk in chunks]
        return chunks_content,chunks_metadata


    def get_used_input(self)->dict:
        faiss_indexer_input = self.faiss_indexer.get_used_input()

        return {
            "faiss_indexer_input": faiss_indexer_input,
            "llm_model": self.llm._identifying_params,
        }
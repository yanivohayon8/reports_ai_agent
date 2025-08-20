from indexer.indexer import FAISSIndexer
from typing import List
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.documents import Document
from needle_agent.needle_prompts import generation_prompt_template


class NeedleAgent():
    
    def __init__(self, faiss_indexer:FAISSIndexer, llm:BaseChatModel) -> None:
        self.faiss_indexer = faiss_indexer
        self.llm = llm

    def answer(self, query:str)->dict:
        context,chunks = self._retrieve_context(query)
        answer = self._generate(context, query)

        chunks_debug_info = self._get_chunks_debug_info(chunks)

        return {
            "answer":answer,
            "chunks":chunks_debug_info
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

    def _get_chunks_debug_info(self,chunks:List[Document])->str:
        return [{"page_content":chunk.page_content,"metadata":chunk.metadata} for chunk in chunks]

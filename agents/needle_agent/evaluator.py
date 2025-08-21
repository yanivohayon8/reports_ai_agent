from core.pdf_reader import read_pdf
from core.text_splitter import get_text_splitter
import random
from langchain_core.language_models import BaseChatModel
from core.api_utils import get_llm_langchain_openai

from agents.needle_agent.needle_prompts import evaluation_Qna_template
import json

class DatasetCreator:
    def __init__(self, llm:BaseChatModel):
        self.llm = llm
    
    @classmethod
    def from_gpt(cls, model:str="gpt-4o-mini",**kwargs):
        llm = get_llm_langchain_openai(model=model,**kwargs)
        
        return cls(llm)

    def create_dataset(self):
        pass

    def create_dataset_from_pdf(self, pdf_path: str):
        pass

    def _create_random_chunk(self,text: str, chunk_size: int, chunk_overlap: int):
        text_splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_text(text)
        random_chunk = random.choice(chunks)
        
        return random_chunk
    
    def _generate_answer_and_question(self,chunk: str)->dict:
        prompt = evaluation_Qna_template.invoke({"chunk":chunk})
        response = self.llm.invoke(prompt)
        response_json = json.loads(response.content)

        return response_json


    


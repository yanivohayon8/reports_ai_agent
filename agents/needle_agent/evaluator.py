from core.pdf_reader import read_pdf
from core.text_splitter import get_text_splitter
import random
from langchain_core.language_models import BaseChatModel
from agents.needle_agent.needle_prompts import evaluation_Qna_template
import json

class DatasetCreator:
    def __init__(self, llm:BaseChatModel, 
                 min_chunk_size: int=400, max_chunk_size: int=1000,
                 min_chunk_overlap: int=100, max_chunk_overlap: int=200):
        self.llm = llm
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.min_chunk_overlap = min_chunk_overlap
        self.max_chunk_overlap = max_chunk_overlap

    def create_dataset(self):
        pass

    def create_dataset_from_text(self, text: str, n_chunks: int=10):
        chunks = self._create_chunks(text)
        random_chunks = self._select_random_chunk(chunks, n_chunks=n_chunks)
        questions_and_answers = self._generate_questions_and_answers(random_chunks)

        return questions_and_answers

    def _create_chunks(self, text: str):
        chunk_size = random.randint(self.min_chunk_size, self.max_chunk_size)
        chunk_overlap = random.randint(self.min_chunk_overlap, self.max_chunk_overlap)
        chunks = self._create_chunks_by_size(text, chunk_size, chunk_overlap)
        return chunks

    def _create_chunks_by_size(self,text: str, chunk_size: int, chunk_overlap: int):
        text_splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_text(text)
        return chunks
    
    def _select_random_chunk(self,chunks: list[str], n_chunks: int)->list[str]:
        random_chunks = random.sample(chunks, n_chunks)
        return random_chunks
    
    def _generate_questions_and_answers(self,chunks: list[str])->list[dict]:
        questions_and_answers = [self._generate_answer_and_question(chunk) for chunk in chunks]
        return questions_and_answers

    def _generate_answer_and_question(self,chunk: str)->dict:
        prompt = evaluation_Qna_template.invoke({"chunk":chunk})
        response = self.llm.invoke(prompt)
        question_and_answer = json.loads(response.content)

        return question_and_answer


    


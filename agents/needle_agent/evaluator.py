from core.pdf_reader import read_pdf
from core.text_splitter import get_text_splitter
import random

class DatasetCreator:
    def __init__(self, output_directory: str):
        self.output_directory = output_directory

    def create_dataset(self):
        pass


    def create_dataset_from_pdf(self, pdf_path: str):
        pass

    def _create_random_chunk(self,text: str, chunk_size: int, chunk_overlap: int):
        text_splitter = get_text_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_text(text)
        
        random_chunk = random.choice(chunks)
        return random_chunk

    


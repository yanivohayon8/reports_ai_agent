import faiss
import os
from pathlib import Path
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from core.api_utils import get_openai_embeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.pdf_reader import read_pdf



class FAISSIndexer():
    """
    A simple FAISS indexer using OpenAI embeddings.
    Supports loading existing databases or creating new ones.
    """
    
    @classmethod
    def from_small_embedding(cls,embedding_model_name:str="text-embedding-3-small",directory_path:str=None, # type: ignore
                             dimension:int=1536):
        embeddings = get_openai_embeddings(model=embedding_model_name,dimensions=dimension)

        return cls(embeddings,directory_path)


    def __init__(self,embedding_model: OpenAIEmbeddings,directory_path:str=None): 
        """
        Initialize the FAISS indexer.
        
        Args:
            directory_path (str): Directory to store/load the vector database
            embedding_model (OpenAIEmbeddings): OpenAI embeddings model
            load_existing (bool): Whether to load an existing vector database
            index_name (str): Name of the index file (without extension)
        """
        self.embedding_model = embedding_model
        self.vector_store = None

        if directory_path is not None and os.path.exists(directory_path):
            self._load_existing_index(directory_path)
        else:
            self._initialize_index()

    def _initialize_index(self):
        index = faiss.IndexFlatL2(len(self.embedding_model.embed_query("hello world")))

        self.vector_store = FAISS(
            embedding_function=self.embedding_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )

    def _load_existing_index(self,directory_path:str):
        self.vector_store = FAISS.load_local(directory_path,self.embedding_model,allow_dangerous_deserialization=True)

    def add_documents(self,documents:List[Document]):
        self.vector_store.add_documents(documents) 

    def save(self,directory_path:str):
        os.makedirs(directory_path,exist_ok=True)
        self.vector_store.save_local(directory_path) 

    def retrieve(self,query:str,**kwargs):
        num_documents = kwargs.get("num_documents",self._get_num_documents(**kwargs))
        return self.vector_store.similarity_search(query,num_documents)
    
    def _get_num_documents(self,**kwargs):
        # TODO: Implement a better way to get the number of documents?
        return 10






class TextChunker():
    @staticmethod
    def get_metadata_keys():
        return  [
                "FileName",
                "PageNumber",
                "ChunkSummary",
                "Keywords",
                "CriticalEntities",
                "IncidentDate", # if relevant
                "SectionType", # if relevant
                "AmountRange", # if relevant
                "FigureId", # if relevant
                "TableId", # if relevant
                "ClientId", # if relevant
                "CaseId" # if relevant
            ]
    
    def __init__(self,text:str,faiss_indexer:FAISSIndexer,chunk_size:int,chunk_overlap:int):
        self.text = text
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.faiss_indexer = faiss_indexer
        
    @classmethod
    def from_pdf(cls,pdf_path:Path,faiss_indexer_directory:Path,chunk_size:int=300,chunk_overlap:int=50):
        text = read_pdf(pdf_path)
        faiss_indexer = FAISSIndexer.from_small_embedding(directory_path=faiss_indexer_directory)
        return cls(text,faiss_indexer,chunk_size,chunk_overlap)
    
    def chunk(self,pdf_path:Path):
        chunks = self._chunk_text()

        for chunk in chunks:
            metadata = {}
            metadata["FileName"] = pdf_path.name
            metadata["PageNumber"] = self._get_page_number(chunk)
            metadata["ChunkSummary"] = self._get_chunk_summary(chunk)
            metadata["Keywords"] = self._get_keywords(chunk)
            metadata["FigureId"] = self._get_figure_id(chunk)

            self.faiss_indexer.add_documents([Document(page_content=chunk,metadata=metadata)])
    
    def _chunk_text(self):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size,chunk_overlap=self.chunk_overlap)
        return text_splitter.split_text(self.text)
    
    def _get_file_name(self,pdf_path:Path):
        return pdf_path.name
    
    def _get_page_number(self,chunk:str):
        raise NotImplementedError("Not implemented")
    
    def _get_chunk_summary(self,chunk:str):
        raise NotImplementedError("Not implemented")
    
    def _get_keywords(self,chunk:str):
        raise NotImplementedError("Not implemented")

    def _get_figure_id(self,chunk:str):
        raise NotImplementedError("Not implemented")
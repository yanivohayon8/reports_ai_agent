import faiss
import os
from pathlib import Path
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.api_utils import get_openai_embeddings
from core.pdf_reader import read_pdf
import json


class FAISSIndexer():
    """
    A simple FAISS indexer using OpenAI embeddings.
    Supports loading existing databases or creating new ones.
    """
    
    @classmethod
    def from_small_embedding(cls,embedding_model_name:str="text-embedding-3-small",directory_path:str=None, # type: ignore
                             dimension:int=1536):
        embeddings = get_openai_embeddings(model=embedding_model_name,dimension=dimension)

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

        if directory_path is not None and self._is_index_exists(directory_path):
            self._load_existing_index(directory_path)
        else:
            self._initialize_index()

        self.metadata = {}
        self.directory_path = directory_path

    def _is_index_exists(self,directory_path:str):
        if not os.path.exists(directory_path):
            return False
        
        for file in os.listdir(directory_path):
            if file.endswith(".faiss"):
                return True
        
        return False


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

    def retrieve(self,query:str,**kwargs):
        num_documents = kwargs.get("num_documents",self._get_num_documents(**kwargs))
        return self.vector_store.similarity_search(query,num_documents)
    
    def _get_num_documents(self,**kwargs):
        # TODO: Implement a better way to get the number of documents?
        return 10

    def save(self,directory_path:str):
        os.makedirs(directory_path,exist_ok=True)
        self.vector_store.save_local(directory_path) 
        self._save_metadata(os.path.join(Path(directory_path),"custom_metadata.json"))

    def _save_metadata(self,file_path:Path):
        with open(file_path,"w") as f:
            json.dump(self.metadata,f)
    
    def _get_metadata_file_path(self)->Path:
        return Path(self.directory_path,"custom_metadata.json")

    def audit_processed_pdf(self,pdf_path:Path):
        self.metadata.setdefault("processed_pdfs",[])
        self.metadata["processed_pdfs"].append(str(pdf_path))

    def audit_splitter(self,text_splitter:RecursiveCharacterTextSplitter):
        self.metadata.setdefault("text_splitter",{})
        self.metadata["text_splitter"]["class_name"] = text_splitter.__class__.__name__

        self.metadata["text_splitter"]["params"] = {
            "chunk_size": text_splitter._chunk_size,
            "chunk_overlap": text_splitter._chunk_overlap,
            "length_function": text_splitter._length_function.__name__,
            "separators": text_splitter._separators
            # TODO: make this more dynamic? 
        }

    def get_used_input(self)->dict:
        self._load_metadata()
        
        return self.metadata

    def _load_metadata(self):
        file_path = self._get_metadata_file_path()

        with open(file_path,"r") as f:
            self.metadata = json.load(f)

class TextChunkerDeprecated():
    
    def __init__(self,faiss_indexer:FAISSIndexer,text_splitter:RecursiveCharacterTextSplitter):
        self.text_splitter = text_splitter
        self.faiss_indexer = faiss_indexer
    
    def chunk(self,pdf_path:Path):
        pages = read_pdf(pdf_path,format="documents")
        chunks = self._chunk_text(pages)

        chunks_doc_processed = []

        for chunk in chunks:
            # TODO: fix this bug - it is the metadata of the whole page, not the chunk
            metadata = chunk.metadata
            metadata["ChunkSummary"] = self._get_chunk_summary(chunk)
            metadata["Keywords"] = self._get_keywords(chunk)
            metadata["FigureId"] = self._get_figure_id(chunk)

            chunks_doc_processed.append(Document(page_content=chunk.page_content,metadata=metadata))

        self.faiss_indexer.add_documents(chunks_doc_processed)
        self.faiss_indexer.audit_processed_pdf(pdf_path)
        self.faiss_indexer.audit_splitter(self.text_splitter)
    
    def _chunk_text(self,pages):
        return self.text_splitter.split_documents(pages)
    
    def _get_chunk_summary(self,chunk:Document):
        # TODO: Implement a way to get the chunk summary (call SummaryAgent Utility?)
        return None
    
    def _get_keywords(self,chunk:Document):
        # TODO: Implement a way to get the keywords
        return None

    def _get_figure_id(self,chunk:Document):
        # TODO: Implement a way to get the figure id 
        return None
    
    def save(self,faiss_indexer_directory:Path):
        self.faiss_indexer.save(faiss_indexer_directory)
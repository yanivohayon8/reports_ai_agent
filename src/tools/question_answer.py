from langchain_core.prompts import ChatPromptTemplate
from src.utils import get_openai_model, get_openai_embedding_model
from langchain_text_splitters import TextSplitter
from langchain_core.documents import Document
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_text_splitters import RecursiveCharacterTextSplitter
    

class QuestionAnswer():

    def __init__(self,text:str,text_splitter:TextSplitter):
        chunks = text_splitter.split_text(text)
        self.documents = [Document(page_content=chunck) for chunck in chunks]
        self.embedding_model = get_openai_embedding_model()

        index = faiss.IndexFlatL2(len(self.embedding_model.embed_query("Hello World")))

        self.vector_store = FAISS(
            embedding_function=self.embedding_model,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={}
        )

        self.vector_store.add_documents(self.documents)

        self.chat_model = get_openai_model()


    def retrieve_context(self,query:str,k:int):
        return self.vector_store.similarity_search(query,k)
    
    def generate(self,context:str,question:str):
        system_message = """You are an expert of quetion answering. Given the following context, answer the question. \\n\\ncontext:{context}\\n\\nquestion:{question}"""

        system_prompt_template = ChatPromptTemplate.from_messages([
            ("system",system_message)
        ])

        prompt = system_prompt_template.invoke({"context":context,"question":question})
        answer = self.chat_model.invoke(prompt)

        return answer 
    
    def _get_k_similar(self,query:str):
        return 5

    def answer_question(self,question:str):
        k = self._get_k_similar(question)
        context = self.retrieve_context(question,k)
        return self.generate(context,question)
    

    @classmethod
    def from_recursive_splitter(cls,text:str):
        # TODO: optimize this
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            length_function=len
        )

        return cls(text,text_splitter)
    

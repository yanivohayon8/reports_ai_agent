from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils import get_openai_model
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document
from langchain_text_splitters import TextSplitter

class MapReduceSummerizer:

    def __init__(self, text, chunk_size=300, chunk_overlap=50):
        self.original_text = text
        self.chat_model = get_openai_model()
        
        # TODO: optimize this
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self.documents = [Document(page_content=text) for text in self.text_splitter.split_text(self.original_text)]

    def get_documents(self):
        return self.documents

    def map(self):
        self.map_prompt = ChatPromptTemplate.from_messages([
            ("system", "Write a concise summary of the following: \\n\\n{context}")
        ])

        mappings = []

        for doc in self.documents:
            map_prompt = self.map_prompt.invoke({"context": doc.page_content})
            summary = self.chat_model.invoke(map_prompt)
            mappings.append(summary)

        return mappings

    def reduce(self,mappings:list[str]):   
        reduce_template ="""
        The following is a set of summaries:
        {docs}
        Take these and distill it into a final, consolidated summary
        of the main themes.
        """
        
        self.reduce_prompt = ChatPromptTemplate([("human",reduce_template)])

        reduce_prompt = self.reduce_prompt.invoke({"docs": mappings})
        return self.chat_model.invoke(reduce_prompt)

    def run(self):
        mappings = self.map()
        return self.reduce(mappings)

class IterativeRefineSummerizer():

    def __init__(self, text, text_splitter:TextSplitter):
        self.original_text = text
        self.chat_model = get_openai_model()

        self.documents = [Document(page_content=text) for text in text_splitter.split_text(self.original_text)]

    @classmethod
    def from_recursive_splitter(cls, text, chunk_size=300, chunk_overlap=50):
        # TODO: optimize this
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        return cls(text, text_splitter)


    def run(self):
        self.summary_prompt = ChatPromptTemplate.from_messages([
            ("system", "Write a concise summary of the following: \\n\\n{context}")
        ])

        summary_prompt = self.summary_prompt.invoke({"context":self.documents[0].page_content})
        summary = self.chat_model.invoke(summary_prompt)

        i = 1
        system_message = """You are an expert at refining summaries. 
                            Given the following summary, refine it with the following context:
                            \\n\\n{summary}\\n\\n{context}
                            """
        self.refine_prompt = ChatPromptTemplate.from_messages([
            ("system", system_message)
        ])

        i = 1
        while i < len(self.documents):
            refine_prompt = self.refine_prompt.invoke({"summary":summary,"context":self.documents[i].page_content})
            summary = self.chat_model.invoke(refine_prompt)
            i+=1

        return summary






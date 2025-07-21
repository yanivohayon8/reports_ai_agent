from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils import get_openai_model
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.documents import Document

class MapReduceSummerize:

    def __init__(self, text, chunk_size=300, chunk_overlap=50):
        self.original_text = text
        self.chat_model = get_openai_model()

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



import pathlib
from core.pdf_reader import read_pdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from summary_agent.prompts import MAP_SUMMARY_PROMPT_CHAT_TEMPLATE,REDUCE_SUMMARY_PROMPT_CHAT_TEMPLATE
from langchain_core.language_models.chat_models import BaseChatModel

class SummaryAgent:

    def __init__(self,text_splitter:RecursiveCharacterTextSplitter,llm:BaseChatModel):
        self.llm = llm
        self.text_splitter = text_splitter

    def summarize_single_pdf(self,pdf_path:pathlib.Path, method:str):
        text = read_pdf(pdf_path,format="text")
        return self.summarize(text,method)

    def summarize(self,text:str,method:str):
        if method == "map_reduce":
            return self._summarize_map_reduce(text)
        elif method == "iterative":
            return self._summarize_iterative_refinement(text)
        else:
            raise ValueError(f"Invalid summary method: {method}")

    def _summarize_map_reduce(self,text:str):
        chunks = self.text_splitter.split_text(text)
        partial_summaries = self._summarize_map(chunks)
        summary = self._summarize_reduce(partial_summaries)
        return summary

    def _summarize_map(self,chunks:list[str]):
        map_chain = MAP_SUMMARY_PROMPT_CHAT_TEMPLATE | self.llm

        partial_summaries = []

        for chunk in chunks:
            response = map_chain.invoke({"text":chunk})
            partial_summaries.append(response.content)

        return partial_summaries
    
    def _summarize_reduce(self,partial_summaries:list[str]):
        reduce_chain = REDUCE_SUMMARY_PROMPT_CHAT_TEMPLATE | self.llm
        combined_text = "\n".join(partial_summaries)
        response = reduce_chain.invoke({"text":combined_text})
        return response.content
    
    def _summarize_iterative_refinement(self,text:str):
        raise NotImplementedError("Iterative refinement summary method not implemented")




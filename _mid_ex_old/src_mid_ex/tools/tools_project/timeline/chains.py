"""Factory functions that return ready LangChain summarization chains."""
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI

from . import prompts

def build_map_reduce_chain(llm: ChatOpenAI):
    """Create a map-reduce summarization chain."""
    return load_summarize_chain(
        llm,
        chain_type="map_reduce",
        map_prompt=prompts.map_prompt(),
        combine_prompt=prompts.reduce_prompt(),
    )

def build_refine_chain(llm: ChatOpenAI):
    """Create a refine summarization chain."""
    return load_summarize_chain(
        llm,
        chain_type="refine",
        question_prompt=prompts.initial_prompt(),
        refine_prompt=prompts.refine_prompt(),
    )

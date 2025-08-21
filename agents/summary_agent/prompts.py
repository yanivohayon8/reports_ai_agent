
from langchain_core.prompts import ChatPromptTemplate

MAP_SUMMARY_PROMPT = """
     "Summarize the following text in a concise way:\n\n{text}"
"""

# TODO: refine the prompt (role-playing, instructions, definitions, maybe some examples)
MAP_SUMMARY_PROMPT_CHAT_TEMPLATE = ChatPromptTemplate.from_template(MAP_SUMMARY_PROMPT)

# TODO: refine the prompt (role-playing, instructions, definitions, maybe some examples)
REDUCE_SUMMARY_PROMPT = """
    "Combine the following partial summaries into a single final summary:\n\n{text}"
"""
REDUCE_SUMMARY_PROMPT_CHAT_TEMPLATE = ChatPromptTemplate.from_template(REDUCE_SUMMARY_PROMPT)

# TODO: refine the prompt (role-playing, instructions, definitions, maybe some examples)
ITERATIVE_REFINEMENT_INITIAL_SUMMARY_PROMPT = """
    "Summarize the following text in a concise way:\n\n{text}"
"""
ITERATIVE_REFINEMENT_INITIAL_SUMMARY_PROMPT_CHAT_TEMPLATE = ChatPromptTemplate.from_template(ITERATIVE_REFINEMENT_INITIAL_SUMMARY_PROMPT)


# TODO: refine the prompt (role-playing, instructions, definitions, maybe some examples)
ITERATIVE_REFINEMENT_PROMPT = """
    "We have an existing summary so far:\n\n{summary}\n\n"
    "Refine it with the following new text, keeping it concise and accurate:\n\n{text}"
"""

ITERATIVE_REFINEMENT_PROMPT_CHAT_TEMPLATE =  ChatPromptTemplate.from_template(ITERATIVE_REFINEMENT_PROMPT)




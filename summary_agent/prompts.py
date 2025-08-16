
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







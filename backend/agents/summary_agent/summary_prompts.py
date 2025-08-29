
from langchain_core.prompts import ChatPromptTemplate

MAP_SUMMARY_PROMPT = """
You are an expert document summarizer. Your task is to create a concise, accurate summary of the provided text.

Guidelines:
- Focus on the main points and key information
- Maintain factual accuracy
- Use clear, professional language
- Include important details like dates, names, and key facts
- Keep the summary concise but comprehensive

Text to summarize:
{document}

Please provide a clear and concise summary:
"""

MAP_SUMMARY_PROMPT_CHAT_TEMPLATE = ChatPromptTemplate.from_template(MAP_SUMMARY_PROMPT)

REDUCE_SUMMARY_PROMPT = """
You are an expert at combining multiple summaries into a coherent, final summary.

Your task is to merge the following partial summaries into a single, well-structured summary that:
- Eliminates redundancy
- Maintains logical flow
- Preserves all important information
- Is concise and readable

Partial summaries to combine:
{partial_summaries}

Please provide a unified, final summary:
"""

REDUCE_SUMMARY_PROMPT_CHAT_TEMPLATE = ChatPromptTemplate.from_template(REDUCE_SUMMARY_PROMPT)

ITERATIVE_REFINEMENT_INITIAL_SUMMARY_PROMPT = """
You are an expert document summarizer. Create an initial summary of the following text.

Guidelines:
- Focus on the main points and key information
- Maintain factual accuracy
- Use clear, professional language
- Include important details like dates, names, and key facts
- Keep the summary concise but comprehensive

Text to summarize:
{document}

Please provide a clear and concise initial summary:
"""

ITERATIVE_REFINEMENT_INITIAL_SUMMARY_PROMPT_CHAT_TEMPLATE = ChatPromptTemplate.from_template(ITERATIVE_REFINEMENT_INITIAL_SUMMARY_PROMPT)

ITERATIVE_REFINEMENT_PROMPT = """
You are an expert at refining summaries with new information.

Current summary:
{summary}

New text to incorporate:
{document}

Please refine the existing summary by:
- Incorporating new relevant information from the text
- Maintaining the existing structure and flow
- Eliminating any redundancy
- Ensuring the summary remains concise and accurate
- Preserving all important details

Refined summary:
"""

ITERATIVE_REFINEMENT_PROMPT_CHAT_TEMPLATE = ChatPromptTemplate.from_template(ITERATIVE_REFINEMENT_PROMPT)

# New prompt for query-based summarization
QUERY_BASED_SUMMARY_PROMPT = """
You are an expert at creating focused summaries based on specific queries.

Query: {query}

Document text to summarize:
{document}

Please create a summary that:
- Directly addresses the user's query
- Focuses on information relevant to the query
- Maintains factual accuracy
- Is concise and well-structured
- Includes key details that answer the query

Summary:
"""

QUERY_BASED_SUMMARY_PROMPT_CHAT_TEMPLATE = ChatPromptTemplate.from_template(QUERY_BASED_SUMMARY_PROMPT)




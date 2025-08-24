from langchain.prompts import PromptTemplate

# Table Question Answering prompt template
TABLE_QA_PROMPT = PromptTemplate(
    input_variables=["query", "table"],
    template=(
        "You are a Table Question Answering expert.\n"
        "The user provided a question and a table.\n"
        "Answer based only on the table content.\n\n"
        "Question: {query}\n\n"
        "Table:\n{table}\n\n"
        "Answer clearly and concisely."
    ),
)

# Alternative prompt for more detailed table analysis
DETAILED_TABLE_QA_PROMPT = PromptTemplate(
    input_variables=["query", "table", "context"],
    template=(
        "You are an expert at analyzing tabular data and answering questions.\n"
        "Given the following question and table, provide a comprehensive answer.\n\n"
        "Question: {query}\n\n"
        "Table:\n{table}\n\n"
        "Additional Context: {context}\n\n"
        "Instructions:\n"
        "1. Analyze the table structure and data\n"
        "2. Identify relevant information for the question\n"
        "3. Provide a clear, accurate answer\n"
        "4. If the question cannot be answered from the table, explain why\n\n"
        "Answer:"
    ),
)

# Simple table extraction prompt
TABLE_EXTRACTION_PROMPT = PromptTemplate(
    input_variables=["text"],
    template=(
        "Extract any tabular data from the following text.\n"
        "If tables are found, format them clearly.\n"
        "If no tables are found, respond with 'No tables detected.'\n\n"
        "Text:\n{text}\n\n"
        "Tables:"
    ),
)

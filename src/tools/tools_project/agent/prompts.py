from langchain.prompts import PromptTemplate

CLASSIFY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=(
        "You are an expert assistant. Decide if the user question asks "
        "for a precise fact (NEEDLE) or a broad overview/synthesis (SUMMARY).\n\n"
        "If NEEDLE, use the QnA tool to answer the question.\n"
        "If SUMMARY, use the Summary tool to provide an overview.\n"
        "Question: {question}\n"
    ),
)

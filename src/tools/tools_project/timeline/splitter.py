"""Split raw text into LangChain Documents."""
from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_into_docs(
    text: str,
    *,
    chunk_size: int = 1_500,
    chunk_overlap: int = 200,
    token_based: bool = False,
) -> List[Document]:
    """Split *text* into document chunks for processing."""
    if token_based:
        splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " "],
        )
    return splitter.create_documents([text])

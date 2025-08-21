from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken, re

def get_text_splitter(**kwargs):
    default_kwargs = {
        "chunk_size":300,
        "chunk_overlap":50,
        "length_function": tiktoken_len,
        "separators": _SEPARATORS
    }
    default_kwargs.update(kwargs)
    
    return RecursiveCharacterTextSplitter(**default_kwargs)

_enc = tiktoken.get_encoding("cl100k_base")

def tiktoken_len(text):
    return len(_enc.encode(text))

_SEPARATORS = [
    "\n## ",   # section headers
    "\n### ",  # subsection headers
    "\n- ",    # bullet points
    "\n* ",    # alt bullet points
    "\n\n",    # paragraphs
    "\n",      # lines
    " ",       # words
    ""         # fallback: characters
]
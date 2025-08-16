import pathlib
from langchain_community.document_loaders import PyPDFLoader

def read_pdf(path: pathlib.Path,format:str="documents"):
    loader = PyPDFLoader(path)
    documents = loader.load()

    if format == "documents":
        return documents
    elif format == "text":
        return "\n\n".join(page.page_content for page in documents)
    else:
        raise ValueError(f"Format {format} not supported")




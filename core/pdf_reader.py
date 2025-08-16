import pathlib
import PyPDF2
from langchain_community.document_loaders import PyPDFLoader


def read_pdf(path: pathlib.Path,loader:str="PyPDFLoader"):
    if loader == "PyPDF2":
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return "\n\n".join(page.extract_text() or "" for page in reader.pages)
    elif loader == "PyPDFLoader":
        loader = PyPDFLoader(path)
        return loader.load()
    else:
        raise ValueError(f"Loader {loader} not supported")


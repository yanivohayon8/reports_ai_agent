import pathlib
try:
    from llama_parse import LlamaParse  # type: ignore
except Exception:
    LlamaParse = None  # Lazy import fallback; functions will guard usage
from .api_utils import verify_llama_parse_api_key
import pandas as pd
from io import StringIO
from typing import List
from langchain_core.documents import Document


def _ensure_llama_available():
    if LlamaParse is None:
        raise ImportError(
            "llama_parse is not installed. Install it and set LLAMA_CLOUD_API_KEY to use pdf_reader."
        )


def read_pdf(path: pathlib.Path, format="documents", split_by_page=True):
    _ensure_llama_available()
    verify_llama_parse_api_key()
    result_type = "markdown"

    if format == "text":
        result_type = "text"

    parser = LlamaParse(
        result_type=result_type,
        extract_charts=True,
        auto_mode=True,
        auto_mode_trigger_on_image_in_page=True,
        auto_mode_trigger_on_table_in_page=True,
        split_by_page=split_by_page,
        output_tables_as_HTML=True,
    )

    llama_documents = []
    extra_info = {"file_name": path.name}

    with open(str(path), "rb") as f:
        llama_documents = parser.load_data(f, extra_info=extra_info)

    if format == "text":
        pages = [page.text for page in llama_documents]
        return "\n\n".join(pages)
    elif format == "documents":
        langchain_documents = []

        for page_index, doc in enumerate(llama_documents):
            metadata = {}
            metadata["source"] = str(path)
            metadata["page"] = page_index + 1

            # NEW: always include tables
            tables = _extract_tables(doc.text)
            metadata["tables"] = tables if tables else []

            langchain_documents.append(Document(page_content=doc.text, metadata=metadata))

        return langchain_documents
    else:
        raise ValueError(f"Format {format} not supported")


def extract_tables(pdf_path: pathlib.Path) -> List[pd.DataFrame]:
    text = _read_pdf_with_html_as_text(pdf_path)
    tables = _extract_tables(text)
    print(f"\tExtracted {len(tables)} tables from {pdf_path}")
    return tables


def _read_pdf_with_html_as_text(pdf_path: pathlib.Path):
    # wrapping function that enforces returning html embeddings
    documents = read_pdf(pdf_path, format="documents", split_by_page=False)
    text = documents[0].page_content
    return text


def _extract_tables(text: str, start_tag="<table>", end_tag="</table>") -> List[pd.DataFrame]:
    if start_tag not in text:
        return []

    tables = []
    start_index = text.find(start_tag)

    while start_index != -1:
        try:
            end_index = text.index(end_tag, start_index)
        except ValueError:
            break

        table_string = text[start_index:end_index + len(end_tag)]
        try:
            dfs = pd.read_html(StringIO(table_string))
            if dfs:
                tables.append(dfs[0])
        except Exception:
            pass

        try:
            start_index = text.index(start_tag, end_index)
        except ValueError:
            break

    return tables

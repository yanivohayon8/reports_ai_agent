import pathlib
import PyPDF2

def read_pdf(path: pathlib.Path) -> str:
    """Extract plain text from a PDF (simple PyPDF2 extraction)."""
    with open(path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        return "\n\n".join(page.extract_text() or "" for page in reader.pages)
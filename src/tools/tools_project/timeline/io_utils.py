"""File I/O utilities – TXT & PDF."""
import pathlib
from typing import Final
import PyPDF2

ENCODINGS: Final = ("utf-8", "cp1252", "latin-1")

def read_file(path: pathlib.Path) -> str:
    """Return file contents or '' if unreadable (caller handles)."""
    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix.lower() == ".pdf":
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            return "".join(page.extract_text() or "" for page in reader.pages)

    for enc in ENCODINGS:
        try:
            return path.read_text(encoding=enc)
        except UnicodeDecodeError:
            continue
    return ""

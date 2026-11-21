from pathlib import Path
from langchain_core.documents import Document
from typing import List
from pypdf import PdfReader


def read_pdf_text(pdf_path: str | Path) -> str:
    """read and return the text content from the pdf files"""
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"File is not found at {path} ")
        
    reader = PdfReader(str(path))
    text_parts=[page.extract_text() or "" for page in reader.pages]
    return "\n".join(text_parts)

def read_multiple_pdfs(pdf_paths:List[str | Path])->List[Document]:
    """read multiple pdfs and return text as List[Document]"""
    docs:List[Document]=[]
    for path in pdf_paths:
        # path_obj=Path(path)
        text=read_pdf_text(path)
        docs.append(Document(page_content=text, metadata={"source": str(path)}))
    return docs

docs_paths=[
    "pdfs/climate_change.pdf",
    "pdfs/global_warming.pdf"
]

# print(read_multiple_pdfs(docs_paths))
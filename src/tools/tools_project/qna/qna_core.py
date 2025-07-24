"""qna_core.py
QnA tool without Parent/Child retriever (simpler, stable).
Includes: MMR, MultiQuery, FAISS distance threshold filter, return_sources.
"""
from __future__ import annotations

import sys
# sys.path.append("../../../../")
sys.path.append("./")


from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from langchain.schema import BaseRetriever, Document
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Multi‑query
from langchain.retrievers.multi_query import MultiQueryRetriever

from src.tools.tools_project.qna.qna_prompts import qa_prompt, multiquery_prompt
from dotenv import load_dotenv
load_dotenv()

from pypdf import PdfReader

import os 

try:
    import tiktoken
except ImportError:  # optional
    tiktoken = None


# ------------------------- Chunking utils ------------------------- #
def _token_len(text: str, model: str = "gpt-4o-mini") -> int:
    if tiktoken is None:
        return len(text)
    enc = tiktoken.encoding_for_model(model)
    return len(enc.encode(text))


def split_into_docs(
    raw_texts: Sequence[str],
    *,
    token_based: bool = True,
    chunk_size: int = 850,
    chunk_overlap: int = 120,
    model_for_tokens: str = "gpt-4o-mini",
) -> List[Document]:
    docs: List[Document] = []

    if token_based and tiktoken is None:
        print("[WARN] token_based=True but tiktoken is missing. Falling back to char split.")
        token_based = False

    if token_based:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=lambda x: _token_len(x, model_for_tokens),
        )
    else:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    for i, txt in enumerate(raw_texts):
        for j, chunk in enumerate(splitter.split_text(txt)):
            # print(chunk)
            docs.append(Document(page_content=chunk, metadata={"source": f"doc_{i}", "chunk_id": j}))
    return docs


# ----------------------------- QnA Tool ---------------------------- #
@dataclass
class QnATool:
    llm: ChatOpenAI
    vectorstore: FAISS
    retriever: BaseRetriever
    qa_chain: RetrievalQA
    return_sources: bool = False 

    def run(self, question: str):
        # Suppress OpenMP runtime error but could make instability,crashed and wrong errors
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
        res = self.qa_chain.invoke({"query": question})
        if self.return_sources:
            print("--------------------------------")
            print(res.get("source_documents", []))
            print("--------------------------------")
            return {"answer": res["result"], "sources": res.get("source_documents", [])}
        return res["result"]


# ------------------------------ Builder ---------------------------- #
def build_qna_tool(
    texts: Sequence[str],
    *,
    # Chunking
    token_based: bool = True,
    chunk_size: int = 850,
    chunk_overlap: int = 120,
    # Embeddings / Vector store
    embedding_model: str = "text-embedding-3-large",
    persist_path: Optional[Path] = None,
    # Retrieval params
    top_k: int = 12,
    fetch_k: int = 50,
    lambda_mult: float = 0.5,
    score_threshold: Optional[float] = None,   # FAISS distance (smaller=better)
    use_multiquery: bool = True,
    n_queries: int = 4,
    # LLM
    llm_model: str = "gpt-4o-mini",
    temperature: float = 0.0,
    # QA
    return_sources: bool = False,
) -> QnATool:
    """Create a fully configured QnATool (no Parent/Child)."""

    embeddings = OpenAIEmbeddings(model=embedding_model)

    # Child docs only
    child_docs = split_into_docs(
        texts,
        token_based=token_based,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        model_for_tokens=llm_model,
    )

    # Vector store
    if persist_path and persist_path.exists():
        vectorstore = FAISS.load_local(str(persist_path), embeddings, allow_dangerous_deserialization=True)
    else:
        vectorstore = FAISS.from_documents(child_docs, embeddings)
        if persist_path:
            vectorstore.save_local(str(persist_path))

    # Base retriever (MMR)
    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
    )

    # Threshold filter (distance <= thr)
    if score_threshold is not None:
        class ThresholdRetriever(BaseRetriever):
            def __init__(self, vs: FAISS, inner: BaseRetriever, thr: float):
                self.vs = vs
                self.inner = inner
                self.thr = thr

            def _get_relevant_documents(self, query: str) -> List[Document]:
                docs_scores = self.vs.similarity_search_with_score(query, k=fetch_k)
                keep = {id(d) for d, s in docs_scores if s <= self.thr}
                ordered = self.inner.get_relevant_documents(query)
                return [d for d in ordered if id(d) in keep]

            async def _aget_relevant_documents(self, query: str) -> List[Document]:
                return self._get_relevant_documents(query)

        retriever: BaseRetriever = ThresholdRetriever(vectorstore, base_retriever, score_threshold)
    else:
        retriever = base_retriever

    # MultiQuery expansion
    if use_multiquery:
        mq_llm = ChatOpenAI(model_name=llm_model, temperature=0)
        prompt = multiquery_prompt().partial(n=str(n_queries))
        retriever = MultiQueryRetriever.from_llm(
            llm=mq_llm,
            retriever=retriever,
            prompt=prompt,
        )

    # QA chain
    llm = ChatOpenAI(model_name=llm_model, temperature=temperature)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": qa_prompt()},
        return_source_documents=return_sources,
    )

    return QnATool(
        llm=llm,
        vectorstore=vectorstore,
        retriever=retriever,
        qa_chain=qa_chain,
        return_sources=return_sources,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple CLI for the QnA tool")
    parser.add_argument("input", nargs="+", help="Path(s) to text or PDF file(s)")
    parser.add_argument("--question", "-q", help="Single question to ask")
    parser.add_argument("--persist", action="store_true", help="Persist FAISS index")
    parser.add_argument("--return-sources", action="store_true")
    args = parser.parse_args()

    def read_file(p: str) -> str:
        ext = Path(p).suffix.lower()
        if ext == ".txt":
            return Path(p).read_text(encoding="utf-8")
        elif ext == ".pdf":
            
            reader = PdfReader(p)
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        else:
            return Path(p).read_text(encoding="utf-8")

    texts = [read_file(p) for p in args.input]

    tool = build_qna_tool(
        texts,
        persist_path=Path("faiss_store") if args.persist else None,
        return_sources=args.return_sources,
    )

    if args.question:
        print(tool.run(args.question))
    else:
        print("Tool built. Use .run(question) in code or pass --question.")

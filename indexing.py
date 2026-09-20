from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_pdf_documents(pdf_path: str | Path):
    loader = PyPDFLoader(str(pdf_path))
    documents = loader.load()
    if not documents:
        raise ValueError("The uploaded PDF could not be loaded or is empty.")
    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = text_splitter.split_documents(documents)
    if not chunks:
        raise ValueError("The uploaded PDF did not produce any text chunks.")
    return chunks


def build_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def create_vectorstore_from_pdf(pdf_path: str | Path):
    documents = load_pdf_documents(pdf_path)
    chunks = split_documents(documents)
    embeddings = build_embeddings()
    return FAISS.from_documents(chunks, embeddings)


if __name__ == "__main__":
    default_pdf = Path("data/module1.pdf")
    vectorstore = create_vectorstore_from_pdf(default_pdf)
    vectorstore.save_local("vectorstore")
    print("FAISS index saved")
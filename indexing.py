from langchain_community.document_loaders import PyPDFLoader

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS

loader = PyPDFLoader("data/module1.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
    )


chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(

model_name = "sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(
    chunks,embeddings
)
vectorstore.save_local("vectorstore")

print("FAISS index saved")
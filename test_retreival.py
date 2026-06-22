from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

db = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

results = db.similarity_search_with_score(
    "What is booting",
    k=3
)

# for doc, score in results:
#     print(score)
#     print(doc.page_content[:200])    



for i,(doc,score) in enumerate(results):
    print(f"\nChunk {i+1} | Distance Score: {score}" )
    print(doc.page_content[:500])


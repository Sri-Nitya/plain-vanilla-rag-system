from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

print("Loading embedding model...")

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print("Loading FAISS index...")

vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

question = input("Enter your question: ")

results = vectorstore.similarity_search(question, k=3)

print("\nTop Retrieved Chunks:\n")

for i, doc in enumerate(results, start=1):
    print(f"========== Chunk {i} ==========")
    print(f"Source: {doc.metadata['source']}")
    print(f"Page: {doc.metadata['page'] + 1}")
    print(doc.page_content[:600])
    print()

print("FAISS loaded successfully!")

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

context = ""

for doc in results:
    context += f"""
Source: {doc.metadata['source']}
Page: {doc.metadata['page'] + 1}

{doc.page_content}
"""
    
prompt = f"""
You are a procurement policy assistant.

Answer ONLY using the provided context with citations to the source documents.

If the answer cannot be found in the context, reply exactly:

"I don't know based on the provided documents."

Context:
{context}

Question:
{question}
"""

response = llm.invoke(prompt)

print("\nAnswer:\n")
print(response.content)
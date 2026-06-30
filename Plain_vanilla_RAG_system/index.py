from langchain_community.document_loaders import PyPDFLoader
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

documents = []

for file in os.listdir("corpus"):
    if file.endswith(".pdf"):

        loader = PyPDFLoader(os.path.join("corpus", file))
        docs = loader.load()

        for doc in docs:
            doc.metadata["source"] = os.path.basename(file)

        documents.extend(docs)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.from_documents(chunks, embeddings)

vectorstore.save_local("faiss_index")

print("Index created successfully!")
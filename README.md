**Plain Vanilla RAG System**

A simple Retrieval-Augmented Generation (RAG) pipeline that answers questions from PDF documents using FAISS for retrieval and Google Gemini for response generation.

**How the Pipeline Works**
Load PDFs

All files from corpus/ are read using PyPDFLoader

Each page is stored with metadata (source, page number)

**Chunking**

Text is split into chunks (500 chars, 100 overlap) using RecursiveCharacterTextSplitter

**Embeddings**

Uses sentence-transformers/all-MiniLM-L6-v2 to convert text into vectors

**Vector Store**
FAISS index is created and saved locally for fast similarity search

Retrieval

User query is embedded and top-3 similar chunks are retrieved

Generation

Retrieved context is passed to Gemini 1.5 Flash

Answer is generated strictly from context with citations

Key Choices

FAISS → fast local vector search

MiniLM embeddings → lightweight and efficient

Small chunks + overlap → better context retention

Strict prompting → reduces hallucinations

Unanswered Questions Handling

If the answer is not present in the retrieved context, the system responds:

"I don't know based on the provided documents."

This is enforced through prompt constraints (no external knowledge allowed).

Future Improvements

Add hybrid search

Add reranking for better chunk selection

Support structured outputs (JSON answers)

Add CSV batch evaluation mode for testing multiple queries

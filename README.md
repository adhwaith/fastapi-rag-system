# FastAPI RAG System (Production Ready)

A Retrieval-Augmented Generation system built with:

- FastAPI
- LangChain
- ChromaDB (Vector Store)
- BAAI BGE Embeddings
- BGE Reranker (CrossEncoder)
- MMR Retrieval
- Ollama (phi3:mini)
- OCR Fallback (pdf2image + pytesseract)
- Streaming Response Support

## Features

- PDF Upload
- Automatic Chunking with Overlap
- Semantic Search
- MMR Retrieval
- Cross-Encoder Reranking
- Streaming LLM Responses
- Persistent Vector Storage
- OCR fallback for scanned PDFs

## Architecture

User → FastAPI → Retriever (MMR) → Reranker → Context → Ollama LLM → Response

## How to Run

1. Install dependencies
2. Install Poppler
3. Install Tesseract
4. Run:
   uvicorn app:app --reload

## Model Used

phi3:mini via Ollama (low-memory optimized)

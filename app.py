import os
import uuid
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import logging

# PDF + OCR
from pdf2image import convert_from_path
import pytesseract
from langchain_community.document_loaders import PyPDFLoader

# Text splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings + VectorDB
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# LLM
from langchain_ollama import OllamaLLM

# -------------------------
# BASIC LOGGING
# -------------------------

logging.basicConfig(level=logging.INFO)

# -------------------------
# CONFIG
# -------------------------

UPLOAD_FOLDER = "uploads"
CHROMA_PATH = "chroma_db"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------
# EMBEDDING MODEL (LIGHT + FAST)
# -------------------------

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5"
)

# -------------------------
# VECTOR STORE
# -------------------------

vectorstore = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings
)

# -------------------------
# LLM (LOW MEMORY CONFIG)
# -------------------------

llm = OllamaLLM(
    model="phi3:mini",
    num_ctx=1024  # reduce RAM usage
)

# -------------------------
# FASTAPI INIT
# -------------------------

app = FastAPI(title="Optimized RAG API")

# -------------------------
# TEXT SPLITTER (Better chunk accuracy)
# -------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=150
)

# -------------------------
# PDF TEXT EXTRACTION
# -------------------------

def extract_text_with_ocr(file_path: str):
    try:
        loader = PyPDFLoader(file_path)
        documents = loader.load()

        if documents and documents[0].page_content.strip():
            return documents
    except:
        pass

    print("Using OCR fallback...")

    images = convert_from_path(
        file_path,
        poppler_path=r"C:\poppler\poppler-25.12.0\Library\bin"
    )

    full_text = ""

    for img in images:
        text = pytesseract.image_to_string(img)
        full_text += text + "\n"

    from langchain_core.documents import Document
    return [Document(page_content=full_text)]


# -------------------------
# BACKGROUND DOCUMENT INGEST
# -------------------------

def process_document(file_path: str):
    logging.info("Processing document...")

    docs = extract_text_with_ocr(file_path)

    chunks = text_splitter.split_documents(docs)

    logging.info(f"Number of chunks: {len(chunks)}")

    vectorstore.add_documents(chunks)

    logging.info("Document stored successfully.")

# -------------------------
# UPLOAD ENDPOINT
# -------------------------

@app.post("/upload")
async def upload_pdf(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    logging.info("Upload started...")

    file_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_FOLDER, f"{file_id}.pdf")

    with open(file_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    logging.info("File saved.")

    background_tasks.add_task(process_document, file_path)

    return {"message": "File uploaded. Processing in background."}

# -------------------------
# REQUEST MODEL
# -------------------------

class QuestionRequest(BaseModel):
    question: str
    k: int = 5
    fetch_k: int = 15

# -------------------------
# ASK ENDPOINT (MMR Retrieval)
# -------------------------

@app.post("/ask")
async def ask_question(request: QuestionRequest):

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": request.k,
            "fetch_k": request.fetch_k
        }
    )

    docs = retriever.invoke(request.question)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
Answer the question using ONLY the context below.
If the answer is not in the context, say "Not found in document."

Context:
{context}

Question:
{request.question}

Answer:
"""

    response = llm.invoke(prompt)

    return {
        "answer": response,
        "sources": len(docs)
    }

# -------------------------
# STREAMING ENDPOINT
# -------------------------

@app.post("/ask-stream")
async def ask_question_stream(request: QuestionRequest):

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": request.k,
            "fetch_k": request.fetch_k
        }
    )

    docs = retriever.invoke(request.question)

    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""
Answer the question using ONLY the context below.
If the answer is not in the context, say "Not found in document."

Context:
{context}

Question:
{request.question}

Answer:
"""

    def stream():
        for chunk in llm.stream(prompt):
            yield chunk

    return StreamingResponse(stream(), media_type="text/plain")

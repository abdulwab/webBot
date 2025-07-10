from fastapi import FastAPI, Request
from app.scraper import scrape_url
from app.chunker import chunk_text
from app.embedder import get_embedding_model
from app.vectordb import create_vector_store, get_retriever
from app.chatbot import generate_answer

app = FastAPI()

# Global retriever cache
retriever = None

@app.post("/init")
async def initialize_from_url(request: Request):
    data = await request.json()
    url = data.get("url")
    text = scrape_url(url)
    docs = chunk_text(text)
    embedder = get_embedding_model()
    vectordb = create_vector_store(docs, embedder)
    global retriever
    retriever = get_retriever(vectordb)
    return {"status": "initialized", "chunks": len(docs)}

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    query = data.get("query")
    if not retriever:
        return {"error": "retriever not initialized. Hit /init first with a URL."}
    answer = generate_answer(query, retriever)
    return {"response": answer}
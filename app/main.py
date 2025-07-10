from fastapi import FastAPI, Request
from app.scraper import scrape_url
from app.chunker import chunk_text
from app.embedder import get_embedding_model
from app.vectordb import create_vector_store, get_vector_store_retriever
from app.chatbot import generate_answer

app = FastAPI(
    title="Ecom AI Agent",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
)

# Global retriever cache
retriever = None


@app.get("/")
async def root():
    return {"status": "Ecom AI Agent is running on Railway 🚀"}


@app.post("/init")
async def initialize_from_url(request: Request):
    data = await request.json()
    url = data.get("url")

    text = scrape_url(url)
    docs = chunk_text(text)
    vectordb = create_vector_store(docs)
    global retriever
    retriever = get_vector_store_retriever()

    return {"status": "initialized", "chunks": len(docs)}


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    query = data.get("query")

    if not retriever:
        return {
            "error": "Retriever not initialized. Call /init first with a URL."
        }

    answer = generate_answer(query, retriever)
    return {"response": answer}


# 👇 Needed for Railway local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
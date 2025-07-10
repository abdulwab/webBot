from fastapi import FastAPI, Request, HTTPException
from app.scraper import scrape_url
from app.chunker import chunk_text
from app.embedder import get_embedding_model
from app.vectordb import create_vector_store, get_vector_store_retriever
from app.chatbot import generate_answer
import traceback
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
    try:
        data = await request.json()
        url = data.get("url")
        
        if not url:
            logger.error("URL parameter is missing")
            raise HTTPException(status_code=400, detail="URL parameter is required")
        
        logger.info(f"Scraping URL: {url}")
        try:
            text = scrape_url(url)
            if not text or len(text) < 10:  # Basic validation for scraped content
                logger.error(f"Failed to scrape content from URL: {url}")
                raise HTTPException(status_code=400, detail="Failed to scrape content from URL or content too short")
        except Exception as e:
            logger.error(f"Error scraping URL {url}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error scraping URL: {str(e)}")
        
        logger.info(f"Chunking text of length {len(text)}")
        try:
            docs = chunk_text(text)
            if not docs:
                logger.error("Chunking resulted in no documents")
                raise HTTPException(status_code=500, detail="Chunking resulted in no documents")
        except Exception as e:
            logger.error(f"Error chunking text: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error chunking text: {str(e)}")
        
        logger.info(f"Creating vector store with {len(docs)} documents")
        try:
            vectordb = create_vector_store(docs)
            global retriever
            retriever = get_vector_store_retriever()
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Error creating vector store: {str(e)}")
        
        logger.info("Successfully initialized retriever")
        return {"status": "initialized", "chunks": len(docs)}
    
    except HTTPException:
        raise
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Unexpected error in /init: {str(e)}\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Initialization failed: {str(e)}")


@app.post("/chat")
async def chat(request: Request):
    try:
        data = await request.json()
        query = data.get("query")
        
        if not query:
            logger.error("Query parameter is missing")
            raise HTTPException(status_code=400, detail="Query parameter is required")

        if not retriever:
            logger.error("Retriever not initialized")
            raise HTTPException(
                status_code=400, 
                detail="Retriever not initialized. Call /init first with a URL."
            )
        
        logger.info(f"Generating answer for query: {query}")
        try:
            answer = generate_answer(query, retriever)
            return {"response": answer}
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}\n{traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")
            
    except HTTPException:
        raise
    except Exception as e:
        error_details = traceback.format_exc()
        logger.error(f"Unexpected error in /chat: {str(e)}\n{error_details}")
        raise HTTPException(status_code=500, detail=f"Chat failed: {str(e)}")


# 👇 Needed for Railway local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
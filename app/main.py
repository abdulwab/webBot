from fastapi import FastAPI, Request, HTTPException
from app.scraper import scrape_url
from app.chunker import chunk_text
from app.embedder import get_embedding_model
from app.vectordb import create_vector_store, get_vector_store_retriever
from app.chatbot import generate_answer
import traceback
import logging
from pydantic import BaseModel, HttpUrl, Field
import hashlib
import os
from typing import Dict, Set, Optional

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

# URL processing cache
# This dictionary will store URLs that have already been processed
processed_urls: Dict[str, int] = {}

# Pydantic models for request validation
class InitRequest(BaseModel):
    url: HttpUrl
    force_refresh: bool = False  # Optional parameter to force reprocessing
    max_pages: int = Field(default=10, ge=1, le=50, description="Maximum number of pages to crawl")
    max_depth: int = Field(default=2, ge=1, le=3, description="Maximum crawl depth")

class InitResponse(BaseModel):
    status: str
    chunks: int
    pages_scraped: int = 0
    cached: bool = False  # Indicates if the response came from cache

class ChatRequest(BaseModel):
    query: str

class ChatResponse(BaseModel):
    answer: str

def get_url_hash(url: str) -> str:
    """Create a hash of the URL to use as a unique identifier."""
    return hashlib.md5(url.encode()).hexdigest()

@app.get("/")
async def root():
    return {"status": "Ecom AI Agent is running on Railway 🚀"}


@app.post("/init", response_model=InitResponse)
async def initialize_from_url(request: InitRequest):
    global retriever
    
    try:
        url = str(request.url)
        url_hash = get_url_hash(url)
        
        # Check if URL has already been processed and force_refresh is False
        if url_hash in processed_urls and not request.force_refresh:
            logger.info(f"URL {url} has already been processed. Using cached retriever.")
            # If we already have a retriever, use it
            if retriever:
                return InitResponse(
                    status="initialized", 
                    chunks=processed_urls[url_hash],
                    pages_scraped=0,  # We don't know how many pages were scraped originally
                    cached=True
                )
            else:
                # If the URL was processed but retriever is None (e.g., after server restart)
                # Get the retriever from the existing vector store
                logger.info("Retriever not in memory. Retrieving from vector store.")
                retriever = get_vector_store_retriever(k=5)
                return InitResponse(
                    status="initialized", 
                    chunks=processed_urls[url_hash],
                    pages_scraped=0,  # We don't know how many pages were scraped originally
                    cached=True
                )
        
        # If URL is new or force_refresh is True, process it
        logger.info(f"Processing URL: {url} with max_pages={request.max_pages}, max_depth={request.max_depth}")
        
        # Scrape content from URL and its subpages
        content = scrape_url(
            url, 
            max_pages=request.max_pages, 
            max_depth=request.max_depth
        )
        
        if not content:
            logger.error(f"Failed to scrape content from URL: {url}")
            raise HTTPException(status_code=400, detail=f"Failed to scrape content from URL: {url}")
        
        # Log a preview of the scraped content
        content_preview = content[:500] + "..." if len(content) > 500 else content
        logger.info(f"Scraped content preview: {content_preview}")
        
        # Count the number of pages scraped (by counting URL markers)
        pages_scraped = content.count("URL: ")
        logger.info(f"Scraped {pages_scraped} pages in total")
        
        # Create chunks from content
        chunks = chunk_text(content)
        if not chunks:
            logger.error("No chunks created from content")
            raise HTTPException(status_code=400, detail="No chunks created from content")
        
        logger.info(f"Created {len(chunks)} chunks from {pages_scraped} pages")
        
        # Create vector store
        vector_store = create_vector_store(chunks)
        if not vector_store:
            logger.error("Failed to create vector store")
            raise HTTPException(status_code=500, detail="Failed to create vector store")
        
        # Get retriever with proper configuration
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5,
                "include_metadata": True
            }
        )
        logger.info("Successfully configured retriever with similarity search")
        
        # Update the processed URLs cache
        processed_urls[url_hash] = len(chunks)
        
        return InitResponse(
            status="initialized", 
            chunks=len(chunks),
            pages_scraped=pages_scraped,
            cached=False
        )
    
    except Exception as e:
        logger.error(f"Error initializing from URL: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error creating vector store: {str(e)}")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    global retriever
    
    try:
        if not retriever:
            logger.error("Retriever not initialized. Call /init endpoint first.")
            raise HTTPException(
                status_code=400, 
                detail="Chatbot not initialized. Please call /init endpoint with a URL first."
            )
        
        query = request.query
        if not query:
            logger.error("Empty query provided")
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Generate answer
        answer = generate_answer(query, retriever)
        if not answer:
            logger.error("Failed to generate answer")
            raise HTTPException(status_code=500, detail="Failed to generate answer")
        
        return ChatResponse(answer=answer)
    
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")


# 👇 Needed for Railway local testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
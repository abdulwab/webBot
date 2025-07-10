import logging
import time
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from app.vectordb import get_vector_store_retriever, init_pinecone
from app.chatbot import generate_answer
from app.scraper import scrape_website
from app.chunker import chunk_text
from app.embedder import get_embedding_model, embed_texts
from app.llm import get_gemini_response

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Website RAG Chatbot API",
    description="API for a chatbot that can answer questions about websites using RAG",
    version="1.0.0",
    docs_url="/docs"
)

# Add CORS middleware
origins = [
    "https://rag-forntend.vercel.app",   # The specific frontend URL with typo
    "https://rag-frontend.vercel.app",    # Corrected URL (in case it's fixed later)
    "http://localhost:3000",             # For local development
    "https://localhost:3000",            # For local development with HTTPS
    "http://127.0.0.1:3000",            # Alternative local development
    "https://127.0.0.1:3000",           # Alternative local development with HTTPS
    # Add any other frontend domains here
]

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://.*\.vercel\.app",  # Allow all Vercel app subdomains
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],  # Allow all headers for simplicity
    max_age=86400,  # Cache preflight requests for 24 hours
)

# Initialize Pinecone at startup
global_retriever = None
try:
    logger.info("Initializing Pinecone at startup")
    init_pinecone()
    
    # Try to initialize the global retriever
    try:
        global_retriever = get_vector_store_retriever(k=5)
        logger.info("Global retriever initialized successfully at startup")
    except Exception as e:
        logger.warning(f"Could not initialize retriever at startup: {str(e)}")
        logger.info("Will try to initialize retriever on first query")
except Exception as e:
    logger.error(f"Error initializing Pinecone at startup: {str(e)}")
    logger.info("Will try to initialize Pinecone again when needed")

# Define request and response models
class WebsiteRequest(BaseModel):
    url: str
    max_pages: Optional[int] = 5
    max_depth: Optional[int] = 1

class QueryRequest(BaseModel):
    query: str

class WebsiteResponse(BaseModel):
    message: str
    pages_processed: int
    chunks_created: int

class QueryResponse(BaseModel):
    answer: str

# Error handler
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"An error occurred: {str(exc)}"}
    )

@app.get("/")
def read_root():
    return {"message": "Welcome to the Website RAG Chatbot API"}

@app.post("/process-website", response_model=WebsiteResponse)
def process_website(request: WebsiteRequest):
    """
    Process a website by scraping content, chunking text, and storing in vector DB
    """
    try:
        start_time = time.time()
        logger.info(f"Processing website: {request.url}")
        
        # Ensure Pinecone is initialized
        try:
            init_pinecone()
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize vector database: {str(e)}"
            )
        
        # Step 1: Scrape website content
        scraped_pages = scrape_website(
            request.url, 
            max_pages=request.max_pages, 
            max_depth=request.max_depth
        )
        
        if not scraped_pages:
            raise HTTPException(status_code=400, detail="Failed to scrape website or no content found")
        
        logger.info(f"Successfully scraped {len(scraped_pages)} pages from {request.url}")
        
        # Step 2: Chunk the text from all pages
        all_chunks = []
        for page_url, page_content in scraped_pages.items():
            chunks = chunk_text(page_content, source=page_url)
            all_chunks.extend(chunks)
            
        if not all_chunks:
            raise HTTPException(status_code=400, detail="Failed to create text chunks from website content")
        
        logger.info(f"Created {len(all_chunks)} chunks from website content")
        
        # Step 3: Get embeddings and store in vector DB
        embedder = get_embedding_model()
        embed_texts(all_chunks, embedder)
        
        processing_time = time.time() - start_time
        logger.info(f"Website processing completed in {processing_time:.2f} seconds")
        
        # Reinitialize the global retriever after adding new data
        global global_retriever
        try:
            global_retriever = get_vector_store_retriever(k=5)
            logger.info("Global retriever reinitialized after adding new data")
        except Exception as e:
            logger.error(f"Error reinitializing retriever: {str(e)}")
            # Continue even if retriever reinitialization fails
        
        return {
            "message": f"Successfully processed website {request.url}",
            "pages_processed": len(scraped_pages),
            "chunks_created": len(all_chunks)
        }
    except Exception as e:
        logger.error(f"Error processing website: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing website: {str(e)}")

@app.post("/query", response_model=QueryResponse)
def query_website(request: QueryRequest):
    """
    Query the chatbot about processed website content
    """
    try:
        logger.info(f"Received query: {request.query}")
        
        # Check if we have a global retriever
        global global_retriever
        if global_retriever is None:
            # Try to initialize it now
            try:
                # Ensure Pinecone is initialized first
                init_pinecone()
                global_retriever = get_vector_store_retriever(k=5)
                logger.info("Global retriever initialized on first query")
            except Exception as e:
                logger.error(f"Error initializing retriever: {str(e)}")
                raise HTTPException(
                    status_code=400, 
                    detail="No website has been processed yet or vector database is not available. Please process a website first."
                )
        
        # Generate answer using the global retriever
        answer = generate_answer(request.query, global_retriever)
        
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)
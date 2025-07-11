import logging
import time
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

from app.vectordb import get_vector_store_retriever, init_pinecone, delete_all_vectors
from app.chatbot import generate_answer
from app.scraper import scrape_website, scrape_single_page
from app.chunker import chunk_text
from app.embedder import get_embedding_model, embed_texts
from app.llm import get_gemini_response

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 2wrap.com specific URLs for comprehensive scraping
TWRAP_IMPORTANT_URLS = [
    "https://2wrap.com/",
    "https://2wrap.com/services",
    "https://2wrap.com/services/",
    "https://2wrap.com/pricing",
    "https://2wrap.com/pricing/",
    "https://2wrap.com/gallery",
    "https://2wrap.com/gallery/",
    "https://2wrap.com/portfolio",
    "https://2wrap.com/portfolio/",
    "https://2wrap.com/about",
    "https://2wrap.com/about/",
    "https://2wrap.com/about-us",
    "https://2wrap.com/contact",
    "https://2wrap.com/contact/",
    "https://2wrap.com/car-wrapping",
    "https://2wrap.com/car-wrapping/",
    "https://2wrap.com/vehicle-wrapping",
    "https://2wrap.com/vehicle-wrapping/",
    "https://2wrap.com/detailing",
    "https://2wrap.com/detailing/",
    "https://2wrap.com/car-detailing",
    "https://2wrap.com/car-detailing/",
    "https://2wrap.com/paint-protection",
    "https://2wrap.com/paint-protection/",
    "https://2wrap.com/ppf",
    "https://2wrap.com/ppf/",
    "https://2wrap.com/ceramic-coating",
    "https://2wrap.com/ceramic-coating/",
    "https://2wrap.com/window-tinting",
    "https://2wrap.com/window-tinting/",
    "https://2wrap.com/colors",
    "https://2wrap.com/colors/",
    "https://2wrap.com/vinyl-colors",
    "https://2wrap.com/vinyl-colors/",
    "https://2wrap.com/process",
    "https://2wrap.com/process/",
    "https://2wrap.com/testimonials",
    "https://2wrap.com/testimonials/",
    "https://2wrap.com/reviews",
    "https://2wrap.com/reviews/",
    "https://2wrap.com/faq",
    "https://2wrap.com/faq/",
    "https://2wrap.com/blog",
    "https://2wrap.com/blog/",
    "https://2wrap.com/location",
    "https://2wrap.com/location/",
    "https://2wrap.com/hours",
    "https://2wrap.com/hours/"
]

def scrape_comprehensive_2wrap() -> Dict[str, str]:
    """
    Comprehensive scraping of 2wrap.com using both automatic discovery and hardcoded URLs
    """
    logger.info("Starting comprehensive scraping of 2wrap.com")
    all_scraped_content = {}
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # First, do automatic discovery with high limits
    try:
        logger.info("Step 1: Automatic discovery scraping with high limits")
        auto_scraped = scrape_website(
            "https://2wrap.com",
            max_retries=3,
            timeout=15,
            max_pages=50,  # Much higher limit
            max_depth=3    # Deeper crawling
        )
        all_scraped_content.update(auto_scraped)
        logger.info(f"Automatic discovery found {len(auto_scraped)} pages")
    except Exception as e:
        logger.warning(f"Automatic discovery failed: {str(e)}, continuing with hardcoded URLs")
    
    # Second, scrape hardcoded important URLs
    logger.info("Step 2: Scraping hardcoded important URLs")
    scraped_hardcoded = 0
    
    for url in TWRAP_IMPORTANT_URLS:
        if url not in all_scraped_content:
            try:
                logger.info(f"Scraping hardcoded URL: {url}")
                content, _ = scrape_single_page(url, headers, timeout=15)
                if content:
                    all_scraped_content[url] = content
                    scraped_hardcoded += 1
                    logger.info(f"Successfully scraped hardcoded URL: {url}")
                else:
                    logger.debug(f"No content from hardcoded URL: {url}")
            except Exception as e:
                logger.debug(f"Failed to scrape hardcoded URL {url}: {str(e)}")
    
    logger.info(f"Hardcoded URL scraping added {scraped_hardcoded} new pages")
    logger.info(f"Total comprehensive scraping result: {len(all_scraped_content)} pages")
    
    if not all_scraped_content:
        raise ValueError("Comprehensive scraping failed to retrieve any content")
    
    return all_scraped_content

# Initialize FastAPI app
app = FastAPI(
    title="2wrap.com RAG Chatbot API",
    description="Comprehensive AI chatbot for 2wrap.com car wrapping and detailing services",
    version="2.0.0",
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
        global_retriever = get_vector_store_retriever(k=10)  # Increased for more context
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
    max_pages: Optional[int] = 20  # Increased default
    max_depth: Optional[int] = 2   # Increased default

class ComprehensiveRequest(BaseModel):
    force_refresh: Optional[bool] = False  # Option to clear existing data

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
    return {"message": "Welcome to the 2wrap.com RAG Chatbot API - Comprehensive car wrapping and detailing assistant"}

@app.get("/status")
def get_status():
    """Get system status and usage instructions"""
    global global_retriever
    
    has_data = global_retriever is not None
    
    return {
        "status": "active",
        "has_vector_data": has_data,
        "instructions": {
            "step_1": "Run POST /process-2wrap-comprehensive to scrape ALL content from 2wrap.com",
            "step_2": "Use POST /query to ask questions - the bot responds as 2wrap in first person",
            "note": "The comprehensive endpoint scrapes 50+ pages including all services, pricing, colors, etc."
        },
        "endpoints": {
            "/process-2wrap-comprehensive": "Scrapes ALL 2wrap.com content (recommended)",
            "/process-website": "Scrapes any website with custom limits",
            "/query": "Ask questions - responds as 2wrap",
            "/status": "This endpoint"
        }
    }

@app.post("/process-2wrap-comprehensive", response_model=WebsiteResponse)
def process_2wrap_comprehensive(request: ComprehensiveRequest):
    """
    Comprehensive processing of 2wrap.com with all services, pricing, colors, and content
    """
    try:
        start_time = time.time()
        logger.info("Starting comprehensive 2wrap.com processing")
        
        # Ensure Pinecone is initialized
        try:
            init_pinecone()
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize vector database: {str(e)}"
            )
        
        # Clear existing data if force refresh is requested
        if request.force_refresh:
            logger.info("Force refresh requested - clearing existing vector data")
            delete_all_vectors()
        
        # Step 1: Comprehensive scraping of 2wrap.com
        scraped_pages = scrape_comprehensive_2wrap()
        
        if not scraped_pages:
            raise HTTPException(status_code=400, detail="Failed to scrape 2wrap.com content")
        
        logger.info(f"Successfully scraped {len(scraped_pages)} pages from 2wrap.com")
        
        # Step 2: Chunk the text from all pages with enhanced metadata
        all_chunks = []
        for page_url, page_content in scraped_pages.items():
            chunks = chunk_text(page_content, source=page_url, use_html_chunking=True)
            all_chunks.extend(chunks)
            
        if not all_chunks:
            raise HTTPException(status_code=400, detail="Failed to create text chunks from website content")
        
        logger.info(f"Created {len(all_chunks)} enhanced chunks from 2wrap.com content")
        
        # Step 3: Get embeddings and store in vector DB with batching
        embedder = get_embedding_model()
        embed_texts(all_chunks, embedder, batch_size=50)  # Smaller batches for stability
        
        processing_time = time.time() - start_time
        logger.info(f"Comprehensive 2wrap.com processing completed in {processing_time:.2f} seconds")
        
        # Reinitialize the global retriever after adding new data
        global global_retriever
        try:
            global_retriever = get_vector_store_retriever(k=10)
            logger.info("Global retriever reinitialized after comprehensive scraping")
        except Exception as e:
            logger.error(f"Error reinitializing retriever: {str(e)}")
            # Continue even if retriever reinitialization fails
        
        return {
            "message": f"Successfully processed comprehensive 2wrap.com content with all services, pricing, and details",
            "pages_processed": len(scraped_pages),
            "chunks_created": len(all_chunks)
        }
    except Exception as e:
        logger.error(f"Error in comprehensive processing: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in comprehensive processing: {str(e)}")

@app.post("/process-website", response_model=WebsiteResponse)
def process_website(request: WebsiteRequest):
    """
    Process a website by scraping content, chunking text, and storing in vector DB
    Enhanced with higher limits for better coverage
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
        
        # Step 1: Scrape website content with enhanced parameters
        scraped_pages = scrape_website(
            request.url, 
            max_pages=request.max_pages, 
            max_depth=request.max_depth,
            max_retries=3,
            timeout=15
        )
        
        if not scraped_pages:
            raise HTTPException(status_code=400, detail="Failed to scrape website or no content found")
        
        logger.info(f"Successfully scraped {len(scraped_pages)} pages from {request.url}")
        
        # Step 2: Chunk the text from all pages with enhanced processing
        all_chunks = []
        for page_url, page_content in scraped_pages.items():
            chunks = chunk_text(page_content, source=page_url, use_html_chunking=True)
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
            global_retriever = get_vector_store_retriever(k=10)
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
    Query the 2wrap.com chatbot - responds as 2wrap in first person
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
                global_retriever = get_vector_store_retriever(k=10)
                logger.info("Global retriever initialized on first query")
            except Exception as e:
                logger.error(f"Error initializing retriever: {str(e)}")
                raise HTTPException(
                    status_code=400, 
                    detail="No website content has been processed yet. Please run comprehensive processing first using /process-2wrap-comprehensive endpoint."
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
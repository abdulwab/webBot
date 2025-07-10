import os
import logging
from typing import List, Optional
from langchain_pinecone import PineconeVectorStore as LangchainPinecone
from langchain_core.documents import Document
import pinecone

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get environment variables
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", "gcp-starter")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "webbot-index")
PINECONE_DIMENSION = 1536  # OpenAI embeddings dimension

# Flag to track if Pinecone has been initialized
pinecone_initialized = False
pinecone_client = None

def init_pinecone():
    """Initialize Pinecone client."""
    global pinecone_initialized, pinecone_client
    
    if pinecone_initialized:
        logger.info("Pinecone already initialized, skipping initialization")
        return
    
    try:
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY environment variable is not set")
            
        # Initialize Pinecone client - handle both old and new API
        logger.info(f"Initializing Pinecone with environment: {PINECONE_ENVIRONMENT}")
        
        # Check which Pinecone client version we're using
        import inspect
        pinecone_init_params = inspect.signature(pinecone.init).parameters
        
        if 'environment' in pinecone_init_params:
            # Old API (pinecone-client < 3.0.0)
            logger.info("Using legacy Pinecone initialization")
            pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
            
            # Check if index exists, create if it doesn't
            existing_indexes = pinecone.list_indexes()
            
            if PINECONE_INDEX_NAME not in existing_indexes:
                logger.info(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
                pinecone.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=PINECONE_DIMENSION,
                    metric="cosine"
                )
                logger.info(f"Successfully created Pinecone index: {PINECONE_INDEX_NAME}")
            else:
                logger.info(f"Pinecone index {PINECONE_INDEX_NAME} already exists")
        else:
            # New API (pinecone-client >= 3.0.0)
            logger.info("Using new Pinecone initialization")
            pinecone_client = pinecone.Pinecone(api_key=PINECONE_API_KEY)
            
            # Check if index exists, create if it doesn't
            indexes = pinecone_client.list_indexes()
            index_names = [idx.name for idx in indexes]
            
            if PINECONE_INDEX_NAME not in index_names:
                logger.info(f"Creating new Pinecone index: {PINECONE_INDEX_NAME}")
                pinecone_client.create_index(
                    name=PINECONE_INDEX_NAME,
                    dimension=PINECONE_DIMENSION,
                    metric="cosine"
                )
                logger.info(f"Successfully created Pinecone index: {PINECONE_INDEX_NAME}")
            else:
                logger.info(f"Pinecone index {PINECONE_INDEX_NAME} already exists")
            
        pinecone_initialized = True
        logger.info("Pinecone initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing Pinecone: {str(e)}")
        raise

def create_vector_store(documents: List[Document]):
    """Create a vector store from documents.
    
    Args:
        documents: List of Document objects with text and metadata
        
    Returns:
        A vector store object
    """
    try:
        # Ensure Pinecone is initialized
        if not pinecone_initialized:
            init_pinecone()
        
        from app.embedder import get_embedding_model
        embedder = get_embedding_model()
        
        logger.info(f"Creating vector store with {len(documents)} documents")
        
        # Create and return the vector store
        vector_store = LangchainPinecone.from_documents(
            documents=documents,
            embedding=embedder,
            index_name=PINECONE_INDEX_NAME
        )
        
        logger.info("Successfully created vector store")
        return vector_store
    except Exception as e:
        logger.error(f"Error creating vector store: {str(e)}")
        raise

def get_vector_store_retriever(k: int = 5):
    """Load existing vector store and return retriever.
    
    Args:
        k: Number of documents to retrieve (default: 5)
        
    Returns:
        A retriever object
    """
    try:
        # Ensure Pinecone is initialized
        if not pinecone_initialized:
            init_pinecone()
            
        logger.info(f"Getting retriever from index {PINECONE_INDEX_NAME} with k={k}")
        from app.embedder import get_embedding_model
        embedder = get_embedding_model()
        
        # Create vector store from existing index
        vector_store = LangchainPinecone.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=embedder
        )
        
        # Configure retriever to fetch k documents
        retriever = vector_store.as_retriever(
            search_type="similarity",  # Use similarity search
            search_kwargs={
                "k": k                # Number of documents to retrieve
            }
        )
        
        logger.info(f"Successfully created retriever with k={k}")
        return retriever
    except Exception as e:
        logger.error(f"Error getting vector store retriever: {str(e)}")
        raise
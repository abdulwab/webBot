import os
import logging
from typing import List, Optional, Dict, Any
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
PINECONE_DIMENSION = 1024  # Match the embedding dimensions

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
                logger.info(f"Successfully created Pinecone index: {PINECONE_INDEX_NAME} with {PINECONE_DIMENSION} dimensions")
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
                logger.info(f"Successfully created Pinecone index: {PINECONE_INDEX_NAME} with {PINECONE_DIMENSION} dimensions")
            else:
                logger.info(f"Pinecone index {PINECONE_INDEX_NAME} already exists")
            
        pinecone_initialized = True
        logger.info("Pinecone initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing Pinecone: {str(e)}")
        raise

def delete_vectors_by_filter(filter_dict: Dict[str, Any]) -> bool:
    """Delete vectors from Pinecone index based on metadata filter.
    
    Args:
        filter_dict: Dictionary of metadata filters to identify vectors to delete
        
    Returns:
        True if deletion was successful, False otherwise
    """
    try:
        # Ensure Pinecone is initialized
        if not pinecone_initialized:
            init_pinecone()
        
        logger.info(f"Deleting vectors with filter: {filter_dict}")
        
        # Get the index
        if pinecone_client:
            # New API
            index = pinecone_client.Index(PINECONE_INDEX_NAME)
        else:
            # Old API
            index = pinecone.Index(PINECONE_INDEX_NAME)
        
        # Delete vectors matching the filter
        delete_response = index.delete(filter=filter_dict)
        logger.info(f"Successfully deleted vectors with filter: {filter_dict}")
        return True
        
    except Exception as e:
        logger.error(f"Error deleting vectors with filter {filter_dict}: {str(e)}")
        return False

def delete_all_vectors() -> bool:
    """Delete all vectors from the Pinecone index.
    
    Returns:
        True if deletion was successful, False otherwise
    """
    try:
        # Ensure Pinecone is initialized
        if not pinecone_initialized:
            init_pinecone()
        
        logger.info("Deleting all vectors from index")
        
        # Get the index
        if pinecone_client:
            # New API
            index = pinecone_client.Index(PINECONE_INDEX_NAME)
        else:
            # Old API
            index = pinecone.Index(PINECONE_INDEX_NAME)
        
        # Delete all vectors
        delete_response = index.delete(delete_all=True)
        logger.info("Successfully deleted all vectors from index")
        return True
        
    except Exception as e:
        logger.error(f"Error deleting all vectors: {str(e)}")
        return False

def delete_vectors_by_source(source_url: str) -> bool:
    """Delete vectors from a specific source URL.
    
    Args:
        source_url: The source URL to delete vectors for
        
    Returns:
        True if deletion was successful, False otherwise
    """
    return delete_vectors_by_filter({"source": source_url})

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

def get_vector_store_retriever(k: int = 5, filter_dict: Optional[Dict[str, Any]] = None):
    """Load existing vector store and return retriever with optional metadata filtering.
    
    Args:
        k: Number of documents to retrieve (default: 5)
        filter_dict: Optional metadata filter dictionary (e.g., {"source": "https://example.com/page"})
        
    Returns:
        A retriever object
    """
    try:
        # Ensure Pinecone is initialized
        if not pinecone_initialized:
            init_pinecone()
            
        logger.info(f"Getting retriever from index {PINECONE_INDEX_NAME} with k={k}")
        if filter_dict:
            logger.info(f"Using metadata filter: {filter_dict}")
        
        from app.embedder import get_embedding_model
        embedder = get_embedding_model()
        
        # Create vector store from existing index
        vector_store = LangchainPinecone.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=embedder
        )
        
        # Configure retriever to fetch k documents with optional filtering
        search_kwargs = {"k": k}
        if filter_dict:
            search_kwargs["filter"] = filter_dict
        
        retriever = vector_store.as_retriever(
            search_type="similarity",  # Use similarity search
            search_kwargs=search_kwargs
        )
        
        logger.info(f"Successfully created retriever with k={k}" + (f" and filter={filter_dict}" if filter_dict else ""))
        return retriever
    except Exception as e:
        logger.error(f"Error getting vector store retriever: {str(e)}")
        raise

def get_vector_store_with_score_threshold(score_threshold: float = 0.7, k: int = 5, filter_dict: Optional[Dict[str, Any]] = None):
    """Load existing vector store and return retriever with similarity score threshold.
    
    Args:
        score_threshold: Minimum similarity score threshold (0.0 to 1.0)
        k: Number of documents to retrieve (default: 5)
        filter_dict: Optional metadata filter dictionary
        
    Returns:
        A retriever object that only returns documents above the score threshold
    """
    try:
        # Ensure Pinecone is initialized
        if not pinecone_initialized:
            init_pinecone()
            
        logger.info(f"Getting score-threshold retriever from index {PINECONE_INDEX_NAME} with threshold={score_threshold}, k={k}")
        if filter_dict:
            logger.info(f"Using metadata filter: {filter_dict}")
        
        from app.embedder import get_embedding_model
        embedder = get_embedding_model()
        
        # Create vector store from existing index
        vector_store = LangchainPinecone.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=embedder
        )
        
        # Configure retriever with similarity score threshold
        search_kwargs = {
            "k": k,
            "score_threshold": score_threshold
        }
        if filter_dict:
            search_kwargs["filter"] = filter_dict
        
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs=search_kwargs
        )
        
        logger.info(f"Successfully created score-threshold retriever with threshold={score_threshold}, k={k}")
        return retriever
    except Exception as e:
        logger.error(f"Error getting score-threshold retriever: {str(e)}")
        raise

def get_existing_sources() -> List[str]:
    """Get all source URLs currently stored in the vector store.
    
    Returns:
        List of source URLs that already exist in the vector store
    """
    try:
        # Ensure Pinecone is initialized
        if not pinecone_initialized:
            init_pinecone()
        
        logger.info("Retrieving existing sources from vector store")
        
        # Get the index
        if pinecone_client:
            # New API
            index = pinecone_client.Index(PINECONE_INDEX_NAME)
        else:
            # Old API
            index = pinecone.Index(PINECONE_INDEX_NAME)
        
        # Query with a dummy vector to get some results and extract metadata
        dummy_vector = [0.0] * PINECONE_DIMENSION
        
        # Get a sample of vectors to extract sources
        query_response = index.query(
            vector=dummy_vector,
            top_k=1000,  # Get up to 1000 results to capture most sources
            include_metadata=True
        )
        
        sources = set()
        for match in query_response.matches:
            if match.metadata and 'source' in match.metadata:
                sources.add(match.metadata['source'])
        
        source_list = list(sources)
        logger.info(f"Found {len(source_list)} unique sources in vector store")
        return source_list
        
    except Exception as e:
        logger.warning(f"Error retrieving existing sources: {str(e)}")
        return []

def check_source_exists(source_url: str) -> bool:
    """Check if a specific source URL exists in the vector store.
    
    Args:
        source_url: The source URL to check
        
    Returns:
        True if the source exists, False otherwise
    """
    try:
        # Ensure Pinecone is initialized
        if not pinecone_initialized:
            init_pinecone()
        
        from app.embedder import get_embedding_model
        embedder = get_embedding_model()
        
        # Create vector store from existing index
        vector_store = LangchainPinecone.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=embedder
        )
        
        # Search for documents with this specific source
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 1,
                "filter": {"source": source_url}
            }
        )
        
        # Try to retrieve documents from this source
        results = retriever.get_relevant_documents("test query")
        exists = len(results) > 0
        
        logger.debug(f"Source {source_url} {'exists' if exists else 'does not exist'} in vector store")
        return exists
        
    except Exception as e:
        logger.debug(f"Error checking if source exists: {str(e)}")
        return False

def get_vector_store_stats() -> Dict[str, Any]:
    """Get statistics about the current vector store content.
    
    Returns:
        Dictionary with statistics about the vector store
    """
    try:
        # Ensure Pinecone is initialized
        if not pinecone_initialized:
            init_pinecone()
        
        # Get the index
        if pinecone_client:
            # New API
            index = pinecone_client.Index(PINECONE_INDEX_NAME)
        else:
            # Old API
            index = pinecone.Index(PINECONE_INDEX_NAME)
        
        # Get index stats
        stats = index.describe_index_stats()
        
        # Get existing sources
        existing_sources = get_existing_sources()
        
        # Count by content type if possible
        content_types = {}
        service_types = {}
        
        # Sample some vectors to analyze metadata
        dummy_vector = [0.0] * PINECONE_DIMENSION
        query_response = index.query(
            vector=dummy_vector,
            top_k=500,  # Sample 500 vectors
            include_metadata=True
        )
        
        for match in query_response.matches:
            if match.metadata:
                # Count content types
                content_type = match.metadata.get('content_type', 'unknown')
                content_types[content_type] = content_types.get(content_type, 0) + 1
                
                # Count service types
                service_list = match.metadata.get('service_types', [])
                if isinstance(service_list, list):
                    for service in service_list:
                        service_types[service] = service_types.get(service, 0) + 1
        
        return {
            "total_vectors": stats.total_vector_count,
            "unique_sources": len(existing_sources),
            "sources": existing_sources,
            "content_types": content_types,
            "service_types": service_types,
            "index_fullness": stats.index_fullness if hasattr(stats, 'index_fullness') else 0.0
        }
        
    except Exception as e:
        logger.error(f"Error getting vector store stats: {str(e)}")
        return {
            "total_vectors": 0,
            "unique_sources": 0,
            "sources": [],
            "content_types": {},
            "service_types": {},
            "index_fullness": 0.0,
            "error": str(e)
        }
import os
import logging
from typing import List, Optional, Dict, Any
from langchain_pinecone import PineconeVectorStore as LangchainPinecone
from langchain_core.documents import Document
import pinecone
import time
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get environment variables
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT", "gcp-starter")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "webbot-index")
PINECONE_DIMENSION = 1024  # Match the embedding dimensions
PINECONE_MAX_PAYLOAD_SIZE = 4 * 1024 * 1024  # 4MB limit

# Flag to track if Pinecone has been initialized
pinecone_initialized = False
pinecone_client = None

def estimate_payload_size(documents: List[Document]) -> int:
    """Estimate the payload size for a batch of documents.
    
    Args:
        documents: List of Document objects
        
    Returns:
        Estimated payload size in bytes
    """
    total_size = 0
    
    for doc in documents:
        # Estimate size of document content
        content_size = len(doc.page_content.encode('utf-8'))
        
        # Estimate size of metadata (rough approximation)
        metadata_size = 0
        if hasattr(doc, 'metadata') and doc.metadata:
            for key, value in doc.metadata.items():
                metadata_size += len(str(key).encode('utf-8'))
                metadata_size += len(str(value).encode('utf-8'))
        
        # Add embedding size (1024 floats * 4 bytes each = 4096 bytes)
        embedding_size = PINECONE_DIMENSION * 4
        
        # Add some overhead for JSON structure
        overhead = 200  # Estimated JSON overhead per document
        
        doc_size = content_size + metadata_size + embedding_size + overhead
        total_size += doc_size
    
    return total_size

def get_optimal_batch_size(documents: List[Document], max_payload_size: int = PINECONE_MAX_PAYLOAD_SIZE) -> int:
    """Calculate optimal batch size to stay under payload limit.
    
    Args:
        documents: List of Document objects
        max_payload_size: Maximum payload size in bytes
        
    Returns:
        Optimal batch size
    """
    if not documents:
        return 50  # Default batch size
    
    # Estimate average document size from first few documents
    sample_size = min(5, len(documents))
    sample_docs = documents[:sample_size]
    estimated_size = estimate_payload_size(sample_docs)
    avg_doc_size = estimated_size / sample_size
    
    # Calculate optimal batch size with safety margin (80% of limit)
    safe_limit = int(max_payload_size * 0.8)
    optimal_batch_size = max(1, int(safe_limit / avg_doc_size))
    
    # Cap at reasonable limits
    optimal_batch_size = min(optimal_batch_size, 100)  # Never exceed 100 docs
    optimal_batch_size = max(optimal_batch_size, 5)    # Never go below 5 docs
    
    logger.info(f"Estimated avg doc size: {avg_doc_size:.0f} bytes, optimal batch size: {optimal_batch_size}")
    return optimal_batch_size

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

def create_vector_store(documents: List[Document], batch_size: Optional[int] = None):
    """Create a vector store from documents with batched uploads to avoid payload size limits.
    
    Args:
        documents: List of Document objects with text and metadata
        batch_size: Number of documents to upload in each batch (auto-calculated if None)
        
    Returns:
        A vector store object
    """
    try:
        # Ensure Pinecone is initialized
        if not pinecone_initialized:
            init_pinecone()
        
        from app.embedder import get_embedding_model
        embedder = get_embedding_model()
        
        # Calculate optimal batch size if not provided
        if batch_size is None:
            batch_size = get_optimal_batch_size(documents)
        
        logger.info(f"Creating vector store with {len(documents)} documents using batch size {batch_size}")
        
        # Create vector store instance first
        vector_store = LangchainPinecone.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=embedder
        )
        
        # Upload documents in batches to avoid payload size limits
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            # Estimate payload size for this batch
            estimated_size = estimate_payload_size(batch_docs)
            size_mb = estimated_size / (1024 * 1024)
            
            logger.info(f"Uploading batch {batch_num}/{total_batches} ({len(batch_docs)} documents, ~{size_mb:.1f}MB)")
            
            # Warn if approaching payload limit
            if estimated_size > PINECONE_MAX_PAYLOAD_SIZE * 0.9:
                logger.warning(f"Batch {batch_num} estimated size ({size_mb:.1f}MB) approaching Pinecone limit (4MB)")
            
            try:
                # Add documents to existing vector store in batches
                vector_store.add_documents(batch_docs)
                logger.info(f"Successfully uploaded batch {batch_num}/{total_batches}")
                
                # Small delay between batches to be respectful to the API
                if batch_num < total_batches:
                    time.sleep(0.5)
                    
            except Exception as batch_error:
                logger.error(f"Error uploading batch {batch_num}: {str(batch_error)}")
                
                # If this batch fails, try with smaller batch size
                if len(batch_docs) > 10:
                    logger.info(f"Retrying batch {batch_num} with smaller chunks (10 documents)")
                    for j in range(0, len(batch_docs), 10):
                        small_batch = batch_docs[j:j + 10]
                        try:
                            vector_store.add_documents(small_batch)
                            logger.info(f"Successfully uploaded small batch {j//10 + 1} from batch {batch_num}")
                        except Exception as small_batch_error:
                            logger.error(f"Failed to upload small batch: {str(small_batch_error)}")
                            # Continue with next small batch
                else:
                    logger.error(f"Failed to upload batch {batch_num} even with small size")
                    # Continue with next batch
        
        logger.info("Successfully created vector store with batched uploads")
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
import os
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone as LangchainPinecone
from app.embedder import get_embedding_model, SimpleOpenAIEmbeddings
from langchain.schema import Document
from pinecone import Pinecone
import logging
import time

logger = logging.getLogger(__name__)

load_dotenv()

# Load from .env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

# Pinecone index dimension
PINECONE_DIMENSION = 1024  # Using 1024 dimensions to match available Pinecone options

# Validate environment variables
if not PINECONE_API_KEY:
    logger.error("PINECONE_API_KEY environment variable is not set")
    raise ValueError("PINECONE_API_KEY environment variable is not set")
if not PINECONE_ENVIRONMENT:
    logger.error("PINECONE_ENVIRONMENT environment variable is not set")
    raise ValueError("PINECONE_ENVIRONMENT environment variable is not set")
if not PINECONE_INDEX_NAME:
    logger.error("PINECONE_INDEX_NAME environment variable is not set")
    raise ValueError("PINECONE_INDEX_NAME environment variable is not set")

try:
    logger.info(f"Initializing Pinecone client with environment: {PINECONE_ENVIRONMENT}")
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
except Exception as e:
    logger.error(f"Failed to initialize Pinecone client: {str(e)}")
    raise


def create_vector_store(documents: list[Document]):
    """Create a new vector store and upsert documents into Pinecone."""
    if not documents:
        logger.error("No documents provided to create_vector_store")
        raise ValueError("No documents provided to create_vector_store")
    
    try:
        embedder = get_embedding_model()
        logger.info(f"Creating/checking Pinecone index: {PINECONE_INDEX_NAME}")
        
        # Check if index exists with retry logic
        max_retries = 3
        retry_count = 0
        index_exists = False
        
        while retry_count < max_retries:
            try:
                index_list = pc.list_indexes()
                if PINECONE_INDEX_NAME not in index_list.names():
                    logger.info(f"Index {PINECONE_INDEX_NAME} does not exist. Creating...")
                    # Create index with 1024 dimensions
                    pc.create_index(name=PINECONE_INDEX_NAME, dimension=PINECONE_DIMENSION, metric="cosine")
                    logger.info(f"Waiting for index {PINECONE_INDEX_NAME} to be ready...")
                    time.sleep(10)  # Wait for index to be ready
                else:
                    logger.info(f"Index {PINECONE_INDEX_NAME} already exists")
                    index_exists = True
                    break
            except Exception as e:
                retry_count += 1
                logger.warning(f"Error checking/creating index (attempt {retry_count}/{max_retries}): {str(e)}")
                if retry_count >= max_retries:
                    logger.error(f"Failed to check/create index after {max_retries} attempts")
                    raise
                time.sleep(2)
        
        logger.info(f"Upserting {len(documents)} documents to Pinecone")
        try:
            # Log document metadata before upserting
            for i, doc in enumerate(documents[:3]):  # Log only first 3 for brevity
                logger.info(f"Document {i+1} metadata: {doc.metadata}")
                preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                logger.info(f"Document {i+1} content preview: {preview}")
            
            # Create vector store with documents
            vector_store = LangchainPinecone.from_documents(
                documents=documents,
                embedding=embedder,
                index_name=PINECONE_INDEX_NAME
            )
            logger.info("Successfully created vector store and upserted documents")
            return vector_store
        except Exception as e:
            logger.error(f"Error upserting documents to Pinecone: {str(e)}")
            raise
            
    except Exception as e:
        logger.error(f"Error in create_vector_store: {str(e)}")
        raise


def get_vector_store_retriever(k: int = 5):
    """Load existing vector store and return retriever.
    
    Args:
        k: Number of documents to retrieve (default: 5)
        
    Returns:
        A retriever object
    """
    try:
        logger.info(f"Getting retriever from index {PINECONE_INDEX_NAME} with k={k}")
        embedder = get_embedding_model()
        
        # Create vector store from existing index
        vector_store = LangchainPinecone.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=embedder
        )
        
        # Configure retriever to return similarity scores and fetch k documents
        retriever = vector_store.as_retriever(
            search_type="similarity",  # Use similarity search
            search_kwargs={
                "k": k,                # Number of documents to retrieve
                "score_threshold": 0.5,  # Minimum similarity score (0-1)
                "include_metadata": True  # Include metadata in results
            }
        )
        
        logger.info(f"Successfully created retriever with k={k}")
        return retriever
    except Exception as e:
        logger.error(f"Error getting vector store retriever: {str(e)}")
        raise
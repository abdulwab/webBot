import os
import logging
import numpy as np
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
TARGET_DIMENSIONS = 1536  # Dimensions for OpenAI embeddings

class CustomOpenAIEmbeddings(Embeddings):
    """Custom wrapper for OpenAI embeddings."""
    
    def __init__(self):
        """Initialize the embeddings model."""
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
            
        self.embeddings_model = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=OPENAI_API_KEY
        )
        logger.info("Initialized OpenAI embeddings model")
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embeddings, one for each text
        """
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts")
            embeddings = self.embeddings_model.embed_documents(texts)
            logger.info(f"Successfully generated embeddings with dimensions: {len(embeddings[0])}")
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embeddings for a query.
        
        Args:
            text: Query text to embed
            
        Returns:
            Embedding for the query
        """
        try:
            logger.info(f"Generating embedding for query: '{text[:50]}...'")
            embedding = self.embeddings_model.embed_query(text)
            logger.info(f"Successfully generated query embedding with dimensions: {len(embedding)}")
            return embedding
        except Exception as e:
            logger.error(f"Error generating query embedding: {str(e)}")
            raise

def get_embedding_model() -> Embeddings:
    """Get the embedding model.
    
    Returns:
        An embedding model instance
    """
    try:
        return CustomOpenAIEmbeddings()
    except Exception as e:
        logger.error(f"Error getting embedding model: {str(e)}")
        raise

def embed_texts(documents: List[Document], embedder: Embeddings = None) -> List[Dict[str, Any]]:
    """Create embeddings for a list of documents and store them in Pinecone.
    
    Args:
        documents: List of Document objects with text and metadata
        embedder: Optional embedder model (will create one if not provided)
        
    Returns:
        List of dictionaries with id, embedding, and metadata
    """
    if not documents:
        logger.error("No documents provided to embed_texts")
        raise ValueError("No documents provided to embed_texts")
        
    try:
        # Get or create embedder
        if not embedder:
            embedder = get_embedding_model()
            
        # Extract text from documents
        texts = [doc.page_content for doc in documents]
        logger.info(f"Creating embeddings for {len(texts)} documents")
        
        # Generate embeddings
        embeddings = embedder.embed_documents(texts)
        
        # Create records for vector store
        records = []
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            records.append({
                "id": f"doc_{i}",
                "embedding": embedding,
                "metadata": doc.metadata,
                "text": doc.page_content
            })
            
        logger.info(f"Successfully created {len(records)} embedding records")
        
        # Import here to avoid circular imports
        from app.vectordb import create_vector_store
        
        # Create vector store with embedded documents
        create_vector_store(documents)
        
        return records
    except Exception as e:
        logger.error(f"Error embedding texts: {str(e)}")
        raise
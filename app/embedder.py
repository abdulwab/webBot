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

class CustomOpenAIEmbeddings(Embeddings):
    """Custom wrapper for OpenAI embeddings using text-embedding-3-small with 1024 dimensions."""
    
    def __init__(self):
        """Initialize the embeddings model."""
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
            
        try:
            # Use text-embedding-3-small with 1024 dimensions
            self.embeddings_model = OpenAIEmbeddings(
                model="text-embedding-3-small",
                dimensions=1024,  # Must match your Pinecone index
                openai_api_key=OPENAI_API_KEY
            )
            logger.info("Initialized OpenAI embeddings model with text-embedding-3-small (1024 dimensions)")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI embeddings model: {str(e)}")
            raise ValueError(f"Could not initialize embedding model: {str(e)}")
    
    def embed_documents(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Generate embeddings for a list of documents with batching support.
        
        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process in each batch
            
        Returns:
            List of embeddings, one for each text
        """
        if not texts:
            logger.warning("Empty texts list provided to embed_documents")
            return []
            
        try:
            logger.info(f"Generating embeddings for {len(texts)} texts with batch size {batch_size}")
            
            # Process in batches to handle large document sets
            all_embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size} ({len(batch)} texts)")
                
                batch_embeddings = self.embeddings_model.embed_documents(batch)
                all_embeddings.extend(batch_embeddings)
                
                # Log the dimensions of the first embedding in the batch for verification
                if batch_embeddings and i == 0:
                    logger.info(f"Embedding dimensions: {len(batch_embeddings[0])}")
            
            logger.info(f"Successfully generated {len(all_embeddings)} embeddings with 1024 dimensions")
            return all_embeddings
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
        if not text:
            logger.warning("Empty text provided to embed_query")
            return []
            
        try:
            logger.info(f"Generating embedding for query: '{text[:50]}...'")
            embedding = self.embeddings_model.embed_query(text)
            
            # Log dimensions for verification
            logger.info(f"Successfully generated query embedding with {len(embedding)} dimensions")
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

def embed_texts(documents: List[Document], embedder: Embeddings = None, batch_size: int = 100) -> List[Dict[str, Any]]:
    """Create embeddings for a list of documents and store them in Pinecone.
    
    Args:
        documents: List of Document objects with text and metadata
        embedder: Optional embedder model (will create one if not provided)
        batch_size: Number of documents to process in each batch
        
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
        logger.info(f"Creating embeddings for {len(texts)} documents with batch size {batch_size}")
        
        # Generate embeddings with batching
        if hasattr(embedder, 'embed_documents') and 'batch_size' in embedder.embed_documents.__code__.co_varnames:
            embeddings = embedder.embed_documents(texts, batch_size=batch_size)
        else:
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
import os
import logging
import openai
from dotenv import load_dotenv
from typing import Any, Dict, List
from langchain.embeddings.base import Embeddings
import numpy as np

logger = logging.getLogger(__name__)

load_dotenv()

class SimpleOpenAIEmbeddings(Embeddings):
    """A simple implementation of OpenAI embeddings using the v0.28.1 client.
    This version resizes the embeddings to 1024 dimensions to match the available Pinecone index option."""
    
    def __init__(
        self,
        model_name: str = "text-embedding-ada-002",
        openai_api_key: str = None,
        target_dimensions: int = 1024,
    ):
        """Initialize with model name, API key, and target dimensions."""
        self.model_name = model_name
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.target_dimensions = target_dimensions
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is not provided and not found in environment")
        
        # Set API key for the openai package
        openai.api_key = self.openai_api_key
        logger.info(f"Initialized SimpleOpenAIEmbeddings with model {self.model_name} (resizing to {self.target_dimensions} dimensions)")
    
    def _resize_embedding(self, embedding: List[float]) -> List[float]:
        """Resize embedding to target dimensions using PCA-like approach."""
        if len(embedding) == self.target_dimensions:
            return embedding
            
        # If original is larger, we'll use a simple dimensionality reduction
        # This is a simple approach - in production you might want a more sophisticated method
        if len(embedding) > self.target_dimensions:
            # Convert to numpy for easier manipulation
            emb_array = np.array(embedding)
            # Take evenly spaced elements to reduce dimensions
            indices = np.round(np.linspace(0, len(embedding) - 1, self.target_dimensions)).astype(int)
            resized = emb_array[indices].tolist()
            return resized
        else:
            # If original is smaller (unlikely), we'll pad with zeros
            return embedding + [0.0] * (self.target_dimensions - len(embedding))
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for multiple documents."""
        if not texts:
            return []
        
        try:
            logger.info(f"Embedding {len(texts)} documents with model {self.model_name}")
            response = openai.Embedding.create(
                model=self.model_name,
                input=texts
            )
            # Resize each embedding to target dimensions
            embeddings = [data["embedding"] for data in response["data"]]
            return [self._resize_embedding(emb) for emb in embeddings]
        except Exception as e:
            logger.error(f"Error embedding documents: {str(e)}")
            raise
    
    def embed_query(self, text: str) -> List[float]:
        """Get embedding for a single query."""
        try:
            logger.info(f"Embedding query with model {self.model_name}")
            response = openai.Embedding.create(
                model=self.model_name,
                input=[text]
            )
            embedding = response["data"][0]["embedding"]
            # Resize to target dimensions
            return self._resize_embedding(embedding)
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise

def get_embedding_model():
    """Create embedding model."""
    try:
        logger.info("Initializing SimpleOpenAIEmbeddings")
        return SimpleOpenAIEmbeddings(
            model_name="text-embedding-ada-002",
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            target_dimensions=1024  # Match the available Pinecone index dimension
        )
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {str(e)}")
        raise
import os
import logging
import openai
from dotenv import load_dotenv
from typing import Any, Dict, List
from langchain.embeddings.base import Embeddings

logger = logging.getLogger(__name__)

load_dotenv()

class SimpleOpenAIEmbeddings(Embeddings):
    """A simple implementation of OpenAI embeddings using the v0.28.1 client."""
    
    def __init__(
        self,
        model_name: str = "text-embedding-ada-002",
        openai_api_key: str = None,
    ):
        """Initialize with model name and API key."""
        self.model_name = model_name
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is not provided and not found in environment")
        
        # Set API key for the openai package
        openai.api_key = self.openai_api_key
        logger.info(f"Initialized SimpleOpenAIEmbeddings with model {self.model_name}")
    
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
            return [data["embedding"] for data in response["data"]]
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
            return response["data"][0]["embedding"]
        except Exception as e:
            logger.error(f"Error embedding query: {str(e)}")
            raise

def get_embedding_model():
    """Create embedding model."""
    try:
        logger.info("Initializing SimpleOpenAIEmbeddings")
        return SimpleOpenAIEmbeddings(
            model_name="text-embedding-ada-002",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    except Exception as e:
        logger.error(f"Failed to initialize embedding model: {str(e)}")
        raise
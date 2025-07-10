from langchain.embeddings import OpenAIEmbeddings
import os
import logging
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv()

def get_embedding_model():
    """Create embedding model using OpenAI's API."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable is not set")
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    try:
        logger.info("Initializing OpenAI embedding model")
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=openai_api_key
        )
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI embedding model: {str(e)}")
        raise
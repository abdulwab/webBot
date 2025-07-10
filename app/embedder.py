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
        # Use minimal parameters to avoid compatibility issues
        return OpenAIEmbeddings(
            model="text-embedding-ada-002",  # Using older model for better compatibility
            openai_api_key=openai_api_key,
            disallowed_special=(),  # Allow special tokens
            chunk_size=1000,  # Process in smaller batches
            client=None  # Let LangChain create the client with default settings
        )
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI embedding model: {str(e)}")
        raise
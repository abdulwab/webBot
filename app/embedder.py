from langchain.embeddings import OpenAIEmbeddings as LangchainOpenAIEmbeddings
import os
import logging
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

load_dotenv()

class CustomOpenAIEmbeddings(LangchainOpenAIEmbeddings):
    """Custom OpenAI embeddings class that avoids problematic parameters."""
    
    def __init__(
        self,
        model: str = "text-embedding-ada-002",
        openai_api_key: Optional[str] = None,
        **kwargs: Any,
    ):
        """Initialize with minimal parameters to avoid compatibility issues."""
        self.model = model
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # Only pass essential parameters to parent class
        super().__init__(
            model=model,
            openai_api_key=self.openai_api_key,
            # Explicitly exclude problematic parameters
            client=None,
        )
        
        # Override client creation to avoid proxies parameter
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.openai_api_key)
        except Exception as e:
            logger.error(f"Failed to create OpenAI client: {str(e)}")
            raise

def get_embedding_model():
    """Create embedding model using custom OpenAI embeddings."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable is not set")
        raise ValueError("OPENAI_API_KEY environment variable is not set")
    
    try:
        logger.info("Initializing custom OpenAI embedding model")
        return CustomOpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=openai_api_key
        )
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI embedding model: {str(e)}")
        raise
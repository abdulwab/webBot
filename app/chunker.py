from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging

logger = logging.getLogger(__name__)

def chunk_text(text):
    """
    Split text into chunks using RecursiveCharacterTextSplitter
    
    Args:
        text: The text to split into chunks
        
    Returns:
        List of Document objects
        
    Raises:
        ValueError: If text is empty or not a string
        Exception: For other chunking errors
    """
    if not text:
        logger.error("Empty text provided to chunk_text")
        raise ValueError("Cannot chunk empty text")
        
    if not isinstance(text, str):
        logger.error(f"Invalid text type provided: {type(text)}")
        raise ValueError(f"Text must be a string, got {type(text)}")
    
    try:
        logger.info(f"Chunking text of length {len(text)}")
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = splitter.create_documents([text])
        
        if not docs:
            logger.warning("Chunking resulted in 0 documents")
        else:
            logger.info(f"Text successfully chunked into {len(docs)} documents")
            
        return docs
    except Exception as e:
        logger.error(f"Error chunking text: {str(e)}")
        raise
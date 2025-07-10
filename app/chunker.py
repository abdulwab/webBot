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
        # Use larger chunks (2000 chars) with more overlap (200 chars) to preserve context
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        docs = splitter.create_documents([text])
        
        if not docs:
            logger.warning("Chunking resulted in 0 documents")
        else:
            logger.info(f"Text successfully chunked into {len(docs)} documents")
            # Log the first few characters of each chunk for debugging
            for i, doc in enumerate(docs[:3]):  # Log only first 3 chunks
                preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                logger.info(f"Chunk {i+1} preview: {preview}")
            
        return docs
    except Exception as e:
        logger.error(f"Error chunking text: {str(e)}")
        raise
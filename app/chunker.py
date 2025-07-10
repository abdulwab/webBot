from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
import re

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
        
        # Split the text by page separators first
        page_separator = "="*50
        pages = text.split(page_separator)
        logger.info(f"Split content into {len(pages)} pages")
        
        # Process each page separately to maintain context
        all_docs = []
        
        # Use larger chunks (2500 chars) with more overlap (250 chars) for multi-page content
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500,
            chunk_overlap=250,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Process each page
        for i, page in enumerate(pages):
            if not page.strip():
                continue
                
            # Create chunks for this page
            page_docs = splitter.create_documents([page])
            
            # Add page number metadata to each chunk
            for doc in page_docs:
                # Extract URL from the page if available
                url_match = re.search(r"URL: (https?://[^\n]+)", doc.page_content)
                if url_match:
                    url = url_match.group(1)
                    # Add metadata
                    doc.metadata["source"] = url
                    doc.metadata["page"] = i
                
            all_docs.extend(page_docs)
            
        if not all_docs:
            logger.warning("Chunking resulted in 0 documents")
        else:
            logger.info(f"Text successfully chunked into {len(all_docs)} documents")
            # Log the first few characters of each chunk for debugging
            for i, doc in enumerate(all_docs[:3]):  # Log only first 3 chunks
                preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                source = doc.metadata.get("source", "unknown")
                logger.info(f"Chunk {i+1} from {source} preview: {preview}")
            
        return all_docs
    except Exception as e:
        logger.error(f"Error chunking text: {str(e)}")
        raise
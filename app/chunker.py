from langchain_text_splitters import RecursiveCharacterTextSplitter, HTMLHeaderTextSplitter
from langchain_core.documents import Document
import logging
import re
from typing import List, Optional, Dict, Any
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Service keywords for car wrapping business metadata
CAR_SERVICES = {
    'wrapping': ['wrap', 'wrapping', 'vinyl', 'film', 'color change', 'custom design', 'vehicle wrap'],
    'detailing': ['detail', 'detailing', 'wash', 'wax', 'polish', 'clean', 'interior', 'exterior'],
    'paint': ['paint', 'coating', 'ceramic', 'protection', 'ppf', 'clear bra'],
    'window': ['tint', 'tinting', 'window', 'glass', 'film'],
    'maintenance': ['maintenance', 'repair', 'restore', 'service'],
    'customization': ['custom', 'personalize', 'design', 'graphics', 'decals', 'lettering']
}

def detect_service_types(text: str) -> List[str]:
    """Detect service types mentioned in the text."""
    text_lower = text.lower()
    detected_services = []
    
    for service_type, keywords in CAR_SERVICES.items():
        for keyword in keywords:
            if keyword in text_lower:
                if service_type not in detected_services:
                    detected_services.append(service_type)
                break
    
    return detected_services

def extract_price_info(text: str) -> Optional[str]:
    """Extract price information from text."""
    # Look for common price patterns
    price_patterns = [
        r'\$\d+(?:,\d{3})*(?:\.\d{2})?',  # $100, $1,000, $1,000.00
        r'\d+\s*dollars?',                # 100 dollars
        r'starting\s+(?:at|from)\s+\$?\d+', # starting at $100
        r'price\s*:\s*\$?\d+',           # price: $100
        r'cost\s*:\s*\$?\d+'             # cost: $100
    ]
    
    for pattern in price_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return ', '.join(matches)
    
    return None

def enhance_metadata_for_car_services(doc: Document) -> Document:
    """Enhance document metadata with car service-specific information."""
    text = doc.page_content
    metadata = doc.metadata.copy()
    
    # Detect service types
    service_types = detect_service_types(text)
    if service_types:
        metadata['service_types'] = service_types
        metadata['primary_service'] = service_types[0]  # First detected service as primary
    
    # Extract price information
    price_info = extract_price_info(text)
    if price_info:
        metadata['price_info'] = price_info
    
    # Determine content type based on URL or content
    source_url = metadata.get('source', '').lower()
    
    if any(keyword in source_url for keyword in ['pricing', 'price', 'cost']):
        metadata['content_type'] = 'pricing'
    elif any(keyword in source_url for keyword in ['service', 'services']):
        metadata['content_type'] = 'services'
    elif any(keyword in source_url for keyword in ['about', 'company']):
        metadata['content_type'] = 'company_info'
    elif any(keyword in source_url for keyword in ['contact', 'location']):
        metadata['content_type'] = 'contact_info'
    elif any(keyword in source_url for keyword in ['gallery', 'portfolio', 'work']):
        metadata['content_type'] = 'portfolio'
    else:
        # Analyze content to determine type
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in ['price', 'cost', '$', 'starting at']):
            metadata['content_type'] = 'pricing'
        elif any(keyword in text_lower for keyword in ['about us', 'our company', 'our story']):
            metadata['content_type'] = 'company_info'
        elif any(keyword in text_lower for keyword in ['contact', 'phone', 'email', 'address']):
            metadata['content_type'] = 'contact_info'
        else:
            metadata['content_type'] = 'general'
    
    # Add business context
    metadata['business_type'] = 'car_wrapping_detailing'
    
    return Document(page_content=text, metadata=metadata)

def chunk_html_by_structure(html_content: str, source: Optional[str] = None) -> List[Document]:
    """Chunk HTML content by structural elements for better granularity."""
    try:
        # Headers to split on - prioritize semantic HTML structure
        headers_to_split_on = [
            ("h1", "Header 1"),
            ("h2", "Header 2"),
            ("h3", "Header 3"),
            ("section", "Section"),
            ("article", "Article"),
            ("div", "Division"),
        ]
        
        # Create HTML header text splitter
        html_splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        
        # Split the HTML content
        html_docs = html_splitter.split_text(html_content)
        
        # Further split large chunks using RecursiveCharacterTextSplitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500,
            chunk_overlap=250,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        final_docs = []
        for doc in html_docs:
            # If the chunk is still too large, split it further
            if len(doc.page_content) > 2500:
                sub_docs = text_splitter.split_documents([doc])
                final_docs.extend(sub_docs)
            else:
                final_docs.append(doc)
        
        # Add source metadata to all chunks
        for doc in final_docs:
            if source:
                doc.metadata["source"] = source
            
            # Enhance with car service metadata
            doc = enhance_metadata_for_car_services(doc)
        
        logger.info(f"HTML structure chunking created {len(final_docs)} documents")
        return final_docs
        
    except Exception as e:
        logger.warning(f"HTML structure chunking failed: {str(e)}, falling back to text chunking")
        return []

def detect_html_content(text: str) -> bool:
    """Detect if content contains meaningful HTML structure."""
    # Look for HTML tags that indicate structure
    html_indicators = ['<h1', '<h2', '<h3', '<section', '<article', '<div', '<p>']
    return any(indicator in text.lower() for indicator in html_indicators)

def chunk_text(text: str, source: Optional[str] = None, use_html_chunking: bool = True) -> List[Document]:
    """
    Split text into chunks using RecursiveCharacterTextSplitter with optional HTML structure awareness
    
    Args:
        text: The text to split into chunks
        source: Optional source URL for the text
        use_html_chunking: Whether to attempt HTML structure-based chunking
        
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
        
        for i, page in enumerate(pages):
            if not page.strip():
                continue
            
            page_docs = []
            
            # Try HTML structure chunking first if enabled and HTML is detected
            if use_html_chunking and detect_html_content(page):
                logger.info(f"Attempting HTML structure chunking for page {i}")
                page_docs = chunk_html_by_structure(page, source)
            
            # If HTML chunking didn't work or wasn't attempted, use text chunking
            if not page_docs:
                logger.info(f"Using text-based chunking for page {i}")
                
                # Use larger chunks (2500 chars) with more overlap (250 chars) for multi-page content
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2500,
                    chunk_overlap=250,
                    separators=["\n\n", "\n", ". ", " ", ""]
                )
                
                # Create chunks for this page
                page_docs = splitter.create_documents([page])
            
            # Process metadata for each chunk
            for doc in page_docs:
                # Extract URL from the page if available and not already set
                if not doc.metadata.get("source"):
                    url_match = re.search(r"URL: (https?://[^\n]+)", doc.page_content)
                    if url_match:
                        url = url_match.group(1)
                        doc.metadata["source"] = url
                    else:
                        # Use provided source if no URL found in content
                        doc.metadata["source"] = source or "unknown"
                
                # Add page number metadata
                doc.metadata["page"] = i
                
                # Enhance with car service-specific metadata
                doc = enhance_metadata_for_car_services(doc)
                
            all_docs.extend(page_docs)
            
        if not all_docs:
            logger.warning("Chunking resulted in 0 documents")
        else:
            logger.info(f"Text successfully chunked into {len(all_docs)} documents")
            
            # Log the first few chunks for debugging with their metadata
            for i, doc in enumerate(all_docs[:3]):  # Log only first 3 chunks
                preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                source = doc.metadata.get("source", "unknown")
                service_types = doc.metadata.get("service_types", [])
                content_type = doc.metadata.get("content_type", "general")
                
                logger.info(f"Chunk {i+1} from {source}")
                logger.info(f"  Content type: {content_type}")
                logger.info(f"  Service types: {service_types}")
                logger.info(f"  Preview: {preview}")
            
        return all_docs
    except Exception as e:
        logger.error(f"Error chunking text: {str(e)}")
        raise

def chunk_with_custom_metadata(documents: List[Document], additional_metadata: Dict[str, Any]) -> List[Document]:
    """Add custom metadata to existing document chunks."""
    enhanced_docs = []
    
    for doc in documents:
        # Create a copy of the document with enhanced metadata
        new_metadata = doc.metadata.copy()
        new_metadata.update(additional_metadata)
        
        enhanced_doc = Document(
            page_content=doc.page_content,
            metadata=new_metadata
        )
        
        # Re-enhance with car service metadata in case new metadata affects detection
        enhanced_doc = enhance_metadata_for_car_services(enhanced_doc)
        enhanced_docs.append(enhanced_doc)
    
    logger.info(f"Enhanced {len(enhanced_docs)} documents with custom metadata")
    return enhanced_docs
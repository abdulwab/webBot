import requests
from bs4 import BeautifulSoup
import logging
from urllib.parse import urlparse
import time
import re

logger = logging.getLogger(__name__)

def is_valid_url(url):
    """Check if URL is valid"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

def clean_text(text):
    """Clean up text by removing extra whitespace and normalizing spaces"""
    # Replace multiple spaces, newlines, tabs with a single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    return text.strip()

def extract_main_content(soup):
    """Extract the main content from the page, focusing on meaningful sections"""
    content = []
    
    # Try to find the main content areas
    main_elements = soup.find_all(['main', 'article', 'div', 'section'])
    
    # Extract text from each heading and paragraph in these elements
    for element in main_elements:
        # Get all headings
        headings = element.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        for heading in headings:
            if heading.get_text(strip=True):
                content.append(f"Heading: {clean_text(heading.get_text())}")
        
        # Get paragraphs
        paragraphs = element.find_all('p')
        for p in paragraphs:
            if p.get_text(strip=True):
                content.append(f"Paragraph: {clean_text(p.get_text())}")
    
    # If we couldn't find structured content, fall back to general text
    if not content:
        logger.warning("Could not find structured content, falling back to general text")
        return soup.get_text(separator=' ', strip=True)
    
    return "\n\n".join(content)

def scrape_url(url, max_retries=2, timeout=10):
    """
    Scrape text content from a URL with retry logic and error handling
    
    Args:
        url: The URL to scrape
        max_retries: Maximum number of retry attempts
        timeout: Request timeout in seconds
    
    Returns:
        Extracted text content or empty string on failure
    """
    if not is_valid_url(url):
        logger.error(f"Invalid URL format: {url}")
        raise ValueError(f"Invalid URL format: {url}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    retry_count = 0
    while retry_count <= max_retries:
        try:
            logger.info(f"Attempting to scrape URL: {url} (Attempt {retry_count + 1}/{max_retries + 1})")
            response = requests.get(url, headers=headers, timeout=timeout)
            
            if response.status_code != 200:
                logger.error(f"Failed to retrieve URL: {url}, Status code: {response.status_code}")
                raise requests.HTTPError(f"HTTP Error: {response.status_code}")
            
            soup = BeautifulSoup(response.content, "html.parser")
            
            # Extract title
            title = soup.title.string if soup.title else "No title found"
            logger.info(f"Page title: {title}")
            
            # Try to extract metadata description
            meta_desc = ""
            meta_tag = soup.find("meta", attrs={"name": "description"})
            if meta_tag and "content" in meta_tag.attrs:
                meta_desc = meta_tag["content"]
                logger.info(f"Found meta description: {meta_desc}")
            
            # Extract main content
            main_content = extract_main_content(soup)
            
            # Combine all the information
            full_content = f"Title: {title}\n\n"
            if meta_desc:
                full_content += f"Description: {meta_desc}\n\n"
            full_content += f"Content:\n{main_content}"
            
            if not full_content or len(full_content) < 50:
                logger.warning(f"Retrieved very little content from {url}")
            else:
                logger.info(f"Successfully scraped {len(full_content)} characters from {url}")
            
            return full_content
            
        except requests.Timeout:
            logger.warning(f"Timeout while scraping {url}")
            retry_count += 1
            if retry_count <= max_retries:
                time.sleep(1)  # Wait before retrying
            else:
                logger.error(f"Max retries reached for {url}")
                raise TimeoutError(f"Timeout while scraping {url} after {max_retries} retries")
                
        except requests.ConnectionError:
            logger.error(f"Connection error while scraping {url}")
            raise ConnectionError(f"Failed to connect to {url}")
            
        except requests.RequestException as e:
            logger.error(f"Request exception while scraping {url}: {str(e)}")
            raise
            
        except Exception as e:
            logger.error(f"Unexpected error while scraping {url}: {str(e)}")
            raise
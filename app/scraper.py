import requests
from bs4 import BeautifulSoup
import logging
from urllib.parse import urlparse
import time

logger = logging.getLogger(__name__)

def is_valid_url(url):
    """Check if URL is valid"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

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
            text = soup.get_text(separator=' ', strip=True)
            
            if not text or len(text) < 10:
                logger.warning(f"Retrieved empty or very short content from {url}")
            else:
                logger.info(f"Successfully scraped {len(text)} characters from {url}")
            
            return text
            
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
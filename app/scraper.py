import requests
from bs4 import BeautifulSoup
import logging
from urllib.parse import urlparse, urljoin
import time
import re
from typing import Set, List, Dict, Optional

logger = logging.getLogger(__name__)

def is_valid_url(url):
    """Check if URL is valid"""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False

def get_domain(url):
    """Extract the domain from a URL"""
    try:
        parsed = urlparse(url)
        return parsed.netloc
    except Exception:
        return None

def is_same_domain(url1, url2):
    """Check if two URLs belong to the same domain"""
    return get_domain(url1) == get_domain(url2)

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

def extract_links(soup, base_url):
    """Extract all links from a page that belong to the same domain"""
    links = set()
    domain = get_domain(base_url)
    
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href']
        # Convert relative URLs to absolute
        full_url = urljoin(base_url, href)
        
        # Only include links from the same domain and skip anchors
        if is_same_domain(full_url, base_url) and '#' not in full_url:
            # Normalize URL by removing trailing slash and fragment
            normalized_url = full_url.rstrip('/')
            normalized_url = normalized_url.split('#')[0]
            links.add(normalized_url)
    
    return links

def scrape_single_page(url, headers, timeout=10):
    """Scrape a single page and return its content and links"""
    try:
        logger.info(f"Scraping page: {url}")
        response = requests.get(url, headers=headers, timeout=timeout)
        
        if response.status_code != 200:
            logger.error(f"Failed to retrieve URL: {url}, Status code: {response.status_code}")
            return None, set()
        
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
        full_content = f"URL: {url}\nTitle: {title}\n\n"
        if meta_desc:
            full_content += f"Description: {meta_desc}\n\n"
        full_content += f"Content:\n{main_content}"
        
        # Extract links for further crawling
        links = extract_links(soup, url)
        
        return full_content, links
    
    except Exception as e:
        logger.error(f"Error scraping {url}: {str(e)}")
        return None, set()

def scrape_url(url, max_retries=2, timeout=10, max_pages=10, max_depth=2):
    """
    Scrape content from a URL and its subpages with depth control
    
    Args:
        url: The starting URL to scrape
        max_retries: Maximum number of retry attempts per page
        timeout: Request timeout in seconds
        max_pages: Maximum number of pages to scrape
        max_depth: Maximum crawl depth
        
    Returns:
        Combined text content from all scraped pages
    """
    if not is_valid_url(url):
        logger.error(f"Invalid URL format: {url}")
        raise ValueError(f"Invalid URL format: {url}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Track visited URLs to avoid duplicates
    visited_urls: Set[str] = set()
    # Queue of URLs to visit with their depth: (url, depth)
    url_queue: List[tuple] = [(url, 0)]
    # Store content from each page
    all_content: List[str] = []
    # Track pages scraped
    pages_scraped = 0
    
    while url_queue and pages_scraped < max_pages:
        current_url, current_depth = url_queue.pop(0)
        
        # Skip if already visited or exceeds max depth
        if current_url in visited_urls or current_depth > max_depth:
            continue
        
        # Mark as visited
        visited_urls.add(current_url)
        
        # Try to scrape with retries
        retry_count = 0
        content = None
        links = set()
        
        while retry_count <= max_retries and not content:
            try:
                content, links = scrape_single_page(current_url, headers, timeout)
                if content:
                    all_content.append(content)
                    pages_scraped += 1
                    logger.info(f"Successfully scraped page {pages_scraped}/{max_pages}: {current_url}")
                    
                    # Add new links to the queue if not at max depth
                    if current_depth < max_depth:
                        for link in links:
                            if link not in visited_urls:
                                url_queue.append((link, current_depth + 1))
                                
                    # Sort queue by depth to do breadth-first search
                    url_queue.sort(key=lambda x: x[1])
                    
                    break
            except Exception as e:
                retry_count += 1
                logger.warning(f"Error on attempt {retry_count} for {current_url}: {str(e)}")
                time.sleep(1)  # Wait before retrying
        
        # Log if all retries failed
        if not content:
            logger.error(f"Failed to scrape {current_url} after {max_retries} attempts")
    
    # Combine all content with page separators
    if not all_content:
        logger.error("No content was scraped from any page")
        raise ValueError("Failed to scrape any content")
    
    combined_content = "\n\n" + "="*50 + "\n\n".join(all_content) + "\n\n" + "="*50 + "\n\n"
    logger.info(f"Successfully scraped {pages_scraped} pages with {len(combined_content)} total characters")
    
    return combined_content

def scrape_website(url, max_retries=2, timeout=10, max_pages=10, max_depth=2) -> Dict[str, str]:
    """
    Scrape content from a website and return a dictionary mapping URLs to their content
    
    Args:
        url: The starting URL to scrape
        max_retries: Maximum number of retry attempts per page
        timeout: Request timeout in seconds
        max_pages: Maximum number of pages to scrape
        max_depth: Maximum crawl depth
        
    Returns:
        Dictionary mapping URLs to their content
    """
    if not is_valid_url(url):
        logger.error(f"Invalid URL format: {url}")
        raise ValueError(f"Invalid URL format: {url}")
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    
    # Track visited URLs to avoid duplicates
    visited_urls: Set[str] = set()
    # Queue of URLs to visit with their depth: (url, depth)
    url_queue: List[tuple] = [(url, 0)]
    # Store content from each page
    page_contents: Dict[str, str] = {}
    # Track pages scraped
    pages_scraped = 0
    
    while url_queue and pages_scraped < max_pages:
        current_url, current_depth = url_queue.pop(0)
        
        # Skip if already visited or exceeds max depth
        if current_url in visited_urls or current_depth > max_depth:
            continue
        
        # Mark as visited
        visited_urls.add(current_url)
        
        # Try to scrape with retries
        retry_count = 0
        content = None
        links = set()
        
        while retry_count <= max_retries and not content:
            try:
                content, links = scrape_single_page(current_url, headers, timeout)
                if content:
                    page_contents[current_url] = content
                    pages_scraped += 1
                    logger.info(f"Successfully scraped page {pages_scraped}/{max_pages}: {current_url}")
                    
                    # Add new links to the queue if not at max depth
                    if current_depth < max_depth:
                        for link in links:
                            if link not in visited_urls:
                                url_queue.append((link, current_depth + 1))
                                
                    # Sort queue by depth to do breadth-first search
                    url_queue.sort(key=lambda x: x[1])
                    
                    break
            except Exception as e:
                retry_count += 1
                logger.warning(f"Error on attempt {retry_count} for {current_url}: {str(e)}")
                time.sleep(1)  # Wait before retrying
        
        # Log if all retries failed
        if not content:
            logger.error(f"Failed to scrape {current_url} after {max_retries} attempts")
    
    # Check if we scraped any pages
    if not page_contents:
        logger.error("No content was scraped from any page")
        raise ValueError("Failed to scrape any content")
    
    logger.info(f"Successfully scraped {pages_scraped} pages")
    return page_contents
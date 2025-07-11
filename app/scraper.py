import requests
from bs4 import BeautifulSoup
import logging
from urllib.parse import urlparse, urljoin, parse_qs, urlunparse
from urllib.robotparser import RobotFileParser
import time
import re
import xml.etree.ElementTree as ET
from typing import Set, List, Dict, Optional
import hashlib
import json
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

# Cache configuration
CACHE_DIR = "app/cache"
CACHE_DURATION_HOURS = 24  # Cache scraped content for 24 hours
ROBOTS_CACHE_DURATION_HOURS = 6  # Cache robots.txt for 6 hours

# Query parameters to ignore when normalizing URLs
IGNORE_QUERY_PARAMS = {
    'utm_source', 'utm_medium', 'utm_campaign', 'utm_term', 'utm_content',
    'ref', 'referrer', 'source', 'fbclid', 'gclid', 'msclkid',
    '_ga', '_gid', '_fbp', '_fbc', 'mc_cid', 'mc_eid'
}

def ensure_cache_dir():
    """Ensure the cache directory exists."""
    if not os.path.exists(CACHE_DIR):
        os.makedirs(CACHE_DIR)
        logger.info(f"Created cache directory: {CACHE_DIR}")

def get_cache_key(url: str) -> str:
    """Generate a cache key for a URL."""
    return hashlib.md5(url.encode()).hexdigest()

def is_cache_valid(cache_file: str, hours: int = CACHE_DURATION_HOURS) -> bool:
    """Check if cache file is still valid."""
    if not os.path.exists(cache_file):
        return False
    
    try:
        cache_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
        return datetime.now() - cache_time < timedelta(hours=hours)
    except Exception:
        return False

def save_to_cache(url: str, data: Dict, cache_type: str = "content"):
    """Save data to cache."""
    try:
        ensure_cache_dir()
        cache_key = get_cache_key(url)
        cache_file = os.path.join(CACHE_DIR, f"{cache_type}_{cache_key}.json")
        
        cache_data = {
            "url": url,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, ensure_ascii=False, indent=2)
        
        logger.debug(f"Saved {cache_type} cache for {url}")
    except Exception as e:
        logger.warning(f"Failed to save cache for {url}: {str(e)}")

def load_from_cache(url: str, cache_type: str = "content", hours: int = CACHE_DURATION_HOURS) -> Optional[Dict]:
    """Load data from cache if valid."""
    try:
        cache_key = get_cache_key(url)
        cache_file = os.path.join(CACHE_DIR, f"{cache_type}_{cache_key}.json")
        
        if is_cache_valid(cache_file, hours):
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            logger.debug(f"Loaded {cache_type} cache for {url}")
            return cache_data.get("data")
    except Exception as e:
        logger.debug(f"Failed to load cache for {url}: {str(e)}")
    
    return None

def normalize_url(url: str) -> str:
    """Normalize URL by removing unnecessary query parameters and fragments."""
    try:
        parsed = urlparse(url)
        
        # Parse query parameters
        query_params = parse_qs(parsed.query)
        
        # Filter out ignored parameters
        filtered_params = {
            k: v for k, v in query_params.items() 
            if k.lower() not in IGNORE_QUERY_PARAMS
        }
        
        # Reconstruct query string
        if filtered_params:
            # Convert back to query string format
            query_pairs = []
            for k, values in filtered_params.items():
                for v in values:
                    query_pairs.append(f"{k}={v}")
            new_query = "&".join(query_pairs)
        else:
            new_query = ""
        
        # Reconstruct URL without fragment and with filtered query
        normalized = urlunparse((
            parsed.scheme,
            parsed.netloc,
            parsed.path.rstrip('/'),
            parsed.params,
            new_query,
            ""  # Remove fragment
        ))
        
        return normalized
    except Exception as e:
        logger.warning(f"Failed to normalize URL {url}: {str(e)}")
        return url

def can_fetch_robots(url: str, user_agent: str = "*") -> bool:
    """Check if URL can be fetched according to robots.txt."""
    try:
        parsed_url = urlparse(url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        robots_url = urljoin(base_url, "/robots.txt")
        
        # Check cache first
        cached_robots = load_from_cache(robots_url, "robots", ROBOTS_CACHE_DURATION_HOURS)
        
        if cached_robots is not None:
            rp = RobotFileParser()
            rp.set_url(robots_url)
            # Set the robots.txt content from cache
            rp.modified()
            if cached_robots.get("content"):
                # Parse cached content
                lines = cached_robots["content"].split('\n')
                rp.read()  # This will try to fetch, but we'll override
                # Unfortunately, RobotFileParser doesn't have a direct way to set content
                # So we'll implement a simple check ourselves
                return simple_robots_check(cached_robots["content"], url, user_agent)
            return True
        
        # Fetch robots.txt
        try:
            response = requests.get(robots_url, timeout=5)
            robots_content = response.text if response.status_code == 200 else ""
            
            # Cache the robots.txt content
            save_to_cache(robots_url, {"content": robots_content}, "robots")
            
            # Check if URL is allowed
            return simple_robots_check(robots_content, url, user_agent)
            
        except Exception as e:
            logger.debug(f"Could not fetch robots.txt from {robots_url}: {str(e)}")
            return True  # If we can't fetch robots.txt, assume it's allowed
            
    except Exception as e:
        logger.warning(f"Error checking robots.txt for {url}: {str(e)}")
        return True  # Default to allowing if there's an error

def simple_robots_check(robots_content: str, url: str, user_agent: str = "*") -> bool:
    """Simple robots.txt parser to check if URL is allowed."""
    if not robots_content:
        return True
    
    try:
        parsed_url = urlparse(url)
        path = parsed_url.path
        
        lines = robots_content.strip().split('\n')
        current_user_agent = None
        disallowed_paths = []
        allowed_paths = []
        applies_to_us = False
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            if line.lower().startswith('user-agent:'):
                ua = line.split(':', 1)[1].strip().lower()
                current_user_agent = ua
                applies_to_us = (ua == '*' or ua == user_agent.lower())
                if applies_to_us:
                    disallowed_paths = []
                    allowed_paths = []
            elif applies_to_us and line.lower().startswith('disallow:'):
                disallow_path = line.split(':', 1)[1].strip()
                if disallow_path:
                    disallowed_paths.append(disallow_path)
            elif applies_to_us and line.lower().startswith('allow:'):
                allow_path = line.split(':', 1)[1].strip()
                if allow_path:
                    allowed_paths.append(allow_path)
        
        # Check if path is explicitly allowed
        for allow_path in allowed_paths:
            if path.startswith(allow_path):
                return True
        
        # Check if path is disallowed
        for disallow_path in disallowed_paths:
            if path.startswith(disallow_path):
                return False
        
        return True
        
    except Exception as e:
        logger.warning(f"Error parsing robots.txt: {str(e)}")
        return True

def fetch_sitemap_urls(base_url: str) -> Set[str]:
    """Fetch URLs from sitemap.xml for faster indexing."""
    sitemap_urls = set()
    
    try:
        parsed_url = urlparse(base_url)
        base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        # Common sitemap locations
        sitemap_locations = [
            urljoin(base_domain, "/sitemap.xml"),
            urljoin(base_domain, "/sitemap_index.xml"),
            urljoin(base_domain, "/sitemaps.xml"),
            urljoin(base_domain, "/robots.txt")  # Check robots.txt for sitemap directives
        ]
        
        for sitemap_url in sitemap_locations:
            try:
                logger.info(f"Checking for sitemap at: {sitemap_url}")
                response = requests.get(sitemap_url, timeout=10)
                
                if response.status_code == 200:
                    content = response.text
                    
                    # If it's robots.txt, look for sitemap directives
                    if sitemap_url.endswith("robots.txt"):
                        for line in content.split('\n'):
                            if line.lower().startswith('sitemap:'):
                                actual_sitemap_url = line.split(':', 1)[1].strip()
                                sitemap_urls.update(parse_sitemap(actual_sitemap_url, base_domain))
                    else:
                        # Parse as XML sitemap
                        sitemap_urls.update(parse_sitemap_content(content, base_domain))
                        
            except Exception as e:
                logger.debug(f"Could not fetch sitemap from {sitemap_url}: {str(e)}")
                continue
                
    except Exception as e:
        logger.warning(f"Error fetching sitemaps for {base_url}: {str(e)}")
    
    if sitemap_urls:
        logger.info(f"Found {len(sitemap_urls)} URLs in sitemaps")
    else:
        logger.info("No sitemap URLs found, will use regular crawling")
    
    return sitemap_urls

def parse_sitemap(sitemap_url: str, base_domain: str) -> Set[str]:
    """Parse a single sitemap XML file."""
    urls = set()
    
    try:
        response = requests.get(sitemap_url, timeout=10)
        if response.status_code == 200:
            urls.update(parse_sitemap_content(response.text, base_domain))
    except Exception as e:
        logger.debug(f"Error parsing sitemap {sitemap_url}: {str(e)}")
    
    return urls

def parse_sitemap_content(xml_content: str, base_domain: str) -> Set[str]:
    """Parse sitemap XML content and extract URLs."""
    urls = set()
    
    try:
        root = ET.fromstring(xml_content)
        
        # Handle sitemap index files
        for sitemap in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap'):
            loc_elem = sitemap.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
            if loc_elem is not None and loc_elem.text:
                # Recursively parse nested sitemaps
                urls.update(parse_sitemap(loc_elem.text, base_domain))
        
        # Handle regular sitemap files
        for url_elem in root.findall('.//{http://www.sitemaps.org/schemas/sitemap/0.9}url'):
            loc_elem = url_elem.find('{http://www.sitemaps.org/schemas/sitemap/0.9}loc')
            if loc_elem is not None and loc_elem.text:
                url = loc_elem.text.strip()
                # Only include URLs from the same domain
                if is_same_domain(url, base_domain):
                    urls.add(normalize_url(url))
                    
    except ET.ParseError as e:
        logger.debug(f"Failed to parse sitemap XML: {str(e)}")
    except Exception as e:
        logger.warning(f"Error parsing sitemap content: {str(e)}")
    
    return urls

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
            normalized_url = normalize_url(full_url)
            links.add(normalized_url)
    
    return links

def scrape_single_page(url, headers, timeout=10):
    """Scrape a single page and return its content and links"""
    try:
        # Check cache first
        cached_data = load_from_cache(url, "content")
        if cached_data:
            logger.info(f"Using cached content for: {url}")
            return cached_data.get("content"), set(cached_data.get("links", []))
        
        # Check robots.txt
        if not can_fetch_robots(url, headers.get('User-Agent', '*')):
            logger.warning(f"Robots.txt disallows fetching: {url}")
            return None, set()
        
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
        
        # Cache the results
        cache_data = {
            "content": full_content,
            "links": list(links)
        }
        save_to_cache(url, cache_data, "content")
        
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
    
    # Normalize the starting URL
    url = normalize_url(url)
    
    # Track visited URLs to avoid duplicates
    visited_urls: Set[str] = set()
    # Queue of URLs to visit with their depth: (url, depth)
    url_queue: List[tuple] = [(url, 0)]
    
    # Try to get URLs from sitemap for faster indexing
    sitemap_urls = fetch_sitemap_urls(url)
    if sitemap_urls:
        # Add sitemap URLs to the queue with depth 0 (prioritize them)
        for sitemap_url in list(sitemap_urls)[:max_pages]:  # Limit to max_pages
            if sitemap_url not in visited_urls:
                url_queue.append((sitemap_url, 0))
    
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
                            normalized_link = normalize_url(link)
                            if normalized_link not in visited_urls:
                                url_queue.append((normalized_link, current_depth + 1))
                                
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
    
    # Normalize the starting URL
    url = normalize_url(url)
    
    # Track visited URLs to avoid duplicates
    visited_urls: Set[str] = set()
    # Queue of URLs to visit with their depth: (url, depth)
    url_queue: List[tuple] = [(url, 0)]
    
    # Try to get URLs from sitemap for faster indexing
    sitemap_urls = fetch_sitemap_urls(url)
    if sitemap_urls:
        # Add sitemap URLs to the queue with depth 0 (prioritize them)
        for sitemap_url in list(sitemap_urls)[:max_pages]:  # Limit to max_pages
            if sitemap_url not in visited_urls:
                url_queue.append((sitemap_url, 0))
    
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
                            normalized_link = normalize_url(link)
                            if normalized_link not in visited_urls:
                                url_queue.append((normalized_link, current_depth + 1))
                                
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
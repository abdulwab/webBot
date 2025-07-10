import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Validate API key
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY environment variable is not set")
    raise ValueError("GOOGLE_API_KEY environment variable is not set")

try:
    logger.info("Configuring Gemini API")
    genai.configure(api_key=GOOGLE_API_KEY)
except Exception as e:
    logger.error(f"Failed to configure Gemini API: {str(e)}")
    raise

def get_gemini_response(prompt, max_retries=2):
    """
    Get a response from the Gemini model
    
    Args:
        prompt: The prompt to send to the model
        max_retries: Maximum number of retry attempts
        
    Returns:
        Generated text response
        
    Raises:
        ValueError: If prompt is empty
        Exception: For API errors or other issues
    """
    if not prompt:
        logger.error("Empty prompt provided to get_gemini_response")
        raise ValueError("Cannot generate response for empty prompt")
    
    retry_count = 0
    last_error = None
    
    while retry_count <= max_retries:
        try:
            logger.info(f"Sending prompt to Gemini (Attempt {retry_count + 1}/{max_retries + 1})")
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(prompt)
            
            if not response or not response.text:
                logger.warning("Received empty response from Gemini")
                retry_count += 1
                if retry_count > max_retries:
                    raise ValueError("Received empty response from Gemini after retries")
                continue
                
            logger.info("Successfully received response from Gemini")
            return response.text.strip()
            
        except Exception as e:
            last_error = e
            logger.warning(f"Error calling Gemini API (Attempt {retry_count + 1}): {str(e)}")
            retry_count += 1
            if retry_count > max_retries:
                logger.error(f"Failed to get response from Gemini after {max_retries} retries")
                raise Exception(f"Failed to get response from Gemini: {str(last_error)}")
    
    # This should not be reached, but just in case
    raise Exception(f"Failed to get response from Gemini after retries: {str(last_error)}")
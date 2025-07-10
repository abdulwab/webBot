import google.generativeai as genai
import os
from dotenv import load_dotenv
import logging
import time

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
    
    # Get available models to check if gemini-pro is available
    try:
        available_models = [m.name for m in genai.list_models()]
        logger.info(f"Available models: {available_models}")
        
        # Find the best Gemini model available
        model_name = None
        for candidate in ["gemini-pro", "gemini-1.0-pro", "gemini-1.5-pro"]:
            for available in available_models:
                if candidate in available:
                    model_name = available
                    break
            if model_name:
                break
                
        if not model_name:
            # Fallback to text-bison if no Gemini model is available
            for available in available_models:
                if "text-bison" in available:
                    model_name = available
                    break
        
        if not model_name:
            logger.error("No suitable generative model found")
            raise ValueError("No suitable generative model found in the available models")
            
        logger.info(f"Selected model: {model_name}")
    except Exception as e:
        logger.warning(f"Error listing models: {str(e)}. Will try with default model name.")
        model_name = "gemini-pro"  # Default fallback
    
    while retry_count <= max_retries:
        try:
            logger.info(f"Sending prompt to model {model_name} (Attempt {retry_count + 1}/{max_retries + 1})")
            model = genai.GenerativeModel(model_name)
            
            # Configure safety settings to be more permissive
            safety_settings = {
                "HARASSMENT": "BLOCK_NONE",
                "HATE": "BLOCK_NONE",
                "SEXUAL": "BLOCK_NONE",
                "DANGEROUS": "BLOCK_NONE",
            }
            
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 1024,
            }
            
            response = model.generate_content(
                prompt,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            if not response or not hasattr(response, 'text'):
                logger.warning("Received invalid response format from model")
                retry_count += 1
                time.sleep(1)  # Add a small delay before retrying
                if retry_count > max_retries:
                    raise ValueError("Received invalid response format after retries")
                continue
                
            logger.info("Successfully received response from model")
            return response.text.strip()
            
        except Exception as e:
            last_error = e
            logger.warning(f"Error calling model API (Attempt {retry_count + 1}): {str(e)}")
            retry_count += 1
            time.sleep(2)  # Add a delay before retrying
            if retry_count > max_retries:
                logger.error(f"Failed to get response from model after {max_retries} retries")
                raise Exception(f"Failed to get response from model: {str(last_error)}")
    
    # This should not be reached, but just in case
    raise Exception(f"Failed to get response from model after retries: {str(last_error)}")
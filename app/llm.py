import google.generativeai as genai
import os
import logging
import time

logger = logging.getLogger(__name__)

# Get API key from environment
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

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
    
    # Preferred models in order of preference
    preferred_models = [
        "gemini-1.5-flash",  # Fast, efficient model
        "gemini-1.5-pro",    # More capable model
        "gemini-pro",        # Original model
    ]
    
    # Get available models
    try:
        logger.info("Listing available models")
        available_models = genai.list_models()
        model_names = [m.name for m in available_models]
        logger.info(f"Available models: {model_names}")
        
        # Find the best available model from our preference list
        model_name = None
        for preferred in preferred_models:
            for available in model_names:
                if preferred in available:
                    model_name = available
                    logger.info(f"Selected model: {model_name}")
                    break
            if model_name:
                break
        
        if not model_name:
            # If none of our preferred models are available, use a default
            logger.warning("No preferred models found, using default model")
            model_name = "models/gemini-1.5-flash"  # Default to the recommended model
        
    except Exception as e:
        logger.warning(f"Error listing models: {str(e)}. Will try with default model.")
        model_name = "models/gemini-1.5-flash"  # Default to the recommended model
        logger.info(f"Using default model: {model_name}")
    
    while retry_count <= max_retries:
        try:
            logger.info(f"Sending prompt to model {model_name} (Attempt {retry_count + 1}/{max_retries + 1})")
            model = genai.GenerativeModel(model_name)
            
            # Configure generation parameters
            generation_config = {
                "temperature": 0.7,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 1024,
            }
            
            # Send the prompt to the model
            response = model.generate_content(
                prompt,
                generation_config=generation_config
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
            
            # If we get an error about the model, try the next one in our preference list
            if "model" in str(e).lower() and retry_count <= len(preferred_models):
                next_model_index = retry_count - 1
                if next_model_index < len(preferred_models):
                    model_name = f"models/{preferred_models[next_model_index]}"
                    logger.info(f"Trying alternative model: {model_name}")
            
            time.sleep(2)  # Add a delay before retrying
            
            if retry_count > max_retries:
                logger.error(f"Failed to get response from model after {max_retries} retries")
                raise Exception(f"Failed to get response from model: {str(last_error)}")
    
    # This should not be reached, but just in case
    raise Exception(f"Failed to get response from model after retries: {str(last_error)}")
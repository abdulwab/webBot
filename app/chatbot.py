from app.llm import get_gemini_response
import logging

logger = logging.getLogger(__name__)

def generate_answer(query, retriever):
    """
    Generate an answer to a query using a retriever and LLM
    
    Args:
        query: The user's question
        retriever: A retriever object that can fetch relevant documents
        
    Returns:
        Generated answer text
        
    Raises:
        ValueError: If query is empty or retriever is invalid
        Exception: For other generation errors
    """
    if not query:
        logger.error("Empty query provided to generate_answer")
        raise ValueError("Cannot generate answer for empty query")
        
    if not retriever:
        logger.error("Invalid retriever provided to generate_answer")
        raise ValueError("Cannot generate answer without a valid retriever")
    
    try:
        logger.info(f"Retrieving relevant documents for query: {query}")
        docs = retriever.get_relevant_documents(query)
        
        if not docs:
            logger.warning("No relevant documents found for query")
            context = "No relevant information found."
        else:
            logger.info(f"Found {len(docs)} relevant documents")
            context = "\n\n".join([doc.page_content for doc in docs])
        
        prompt = f"""You are an assistant for an e-commerce store.
Use the context below to answer the user's question.

Context:
{context}

Question:
{query}

Answer:"""
        
        logger.info("Sending prompt to LLM")
        response = get_gemini_response(prompt)
        logger.info("Successfully generated response")
        return response
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        raise
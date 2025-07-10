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
        # Increase the number of relevant documents to retrieve
        docs = retriever.get_relevant_documents(query, k=5)
        
        if not docs:
            logger.warning("No relevant documents found for query")
            context = "No relevant information found."
        else:
            logger.info(f"Found {len(docs)} relevant documents")
            # Format the context with numbered sections and source information
            context_parts = []
            for i, doc in enumerate(docs):
                # Extract source information from metadata
                source = doc.metadata.get("source", "Unknown source")
                source_info = f"Source: {source}"
                
                # Add the document with its source
                context_parts.append(f"[Document {i+1}] {source_info}\n{doc.page_content}")
            context = "\n\n".join(context_parts)
        
        prompt = f"""You are an AI assistant for a website. You need to answer the user's question based ONLY on the provided context.

CONTEXT:
{context}

USER QUESTION:
{query}

INSTRUCTIONS:
1. Answer the question based ONLY on the information in the context.
2. If the context doesn't contain enough information to answer the question fully, say so clearly.
3. Be specific and detailed in your answer, citing the relevant parts of the context.
4. If the context contains information about what the website is about, its products, services, or purpose, include that in your answer.
5. If relevant, mention which specific page or section of the website the information comes from.
6. Do not make up information or use your general knowledge outside of what's in the context.
7. Format your answer in a clear, concise way.

YOUR ANSWER:"""
        
        logger.info("Sending prompt to LLM")
        response = get_gemini_response(prompt)
        logger.info("Successfully generated response")
        return response
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        raise
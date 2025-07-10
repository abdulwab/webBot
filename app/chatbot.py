from app.llm import get_gemini_response
import logging
import time

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
        # Step 1: Vector search to find relevant documents
        logger.info(f"RAG Step 1: Retrieving relevant documents for query: '{query}'")
        logger.info("Converting query to embedding and searching vector store...")
        start_time = time.time()
        
        # Get relevant documents from vector store
        docs = retriever.get_relevant_documents(query)
        
        retrieval_time = time.time() - start_time
        logger.info(f"Vector search completed in {retrieval_time:.2f} seconds")
        
        if not docs:
            logger.warning("No relevant documents found in vector store for this query")
            context = "No relevant information found in the website content."
        else:
            logger.info(f"Found {len(docs)} relevant documents from vector store")
            
            # Log similarity scores if available
            for i, doc in enumerate(docs):
                source = doc.metadata.get("source", "Unknown source")
                score = doc.metadata.get("score", "Unknown score")
                logger.info(f"Document {i+1} from {source} with score: {score}")
                
                # Log a preview of each document
                preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                logger.info(f"Document {i+1} preview: {preview}")
            
            # Format the context with numbered sections and source information
            context_parts = []
            for i, doc in enumerate(docs):
                # Extract source information from metadata
                source = doc.metadata.get("source", "Unknown source")
                source_info = f"Source: {source}"
                
                # Add the document with its source
                context_parts.append(f"[Document {i+1}] {source_info}\n{doc.page_content}")
            context = "\n\n".join(context_parts)
        
        # Step 2: Generate answer using LLM with retrieved context
        logger.info("RAG Step 2: Generating answer using LLM with retrieved context")
        
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
        start_time = time.time()
        response = get_gemini_response(prompt)
        generation_time = time.time() - start_time
        
        logger.info(f"LLM response generated in {generation_time:.2f} seconds")
        logger.info("RAG process completed successfully")
        
        return response
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        raise
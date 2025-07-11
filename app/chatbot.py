from app.llm import get_gemini_response
from app.vectordb import get_vector_store_with_score_threshold
import logging
import time
from typing import List, Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)

# Configuration for confidence scoring
DEFAULT_CONFIDENCE_THRESHOLD = 0.7  # Minimum similarity score to include documents
MIN_DOCUMENTS_FOR_ANSWER = 1        # Minimum number of documents needed to generate an answer
MAX_DOCUMENTS_FOR_CONTEXT = 5       # Maximum documents to include in context

def calculate_confidence_score(documents_with_scores: List[Tuple], threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> Dict[str, Any]:
    """Calculate overall confidence score for the retrieved documents."""
    if not documents_with_scores:
        return {
            "overall_confidence": 0.0,
            "high_confidence_docs": 0,
            "avg_score": 0.0,
            "meets_threshold": False
        }
    
    # Extract scores (handling both (doc, score) tuples and plain documents)
    scores = []
    for item in documents_with_scores:
        if isinstance(item, tuple) and len(item) == 2:
            _, score = item
            if score is not None:
                scores.append(score)
    
    if not scores:
        # If no scores available, assume medium confidence
        return {
            "overall_confidence": 0.5,
            "high_confidence_docs": len(documents_with_scores),
            "avg_score": 0.5,
            "meets_threshold": len(documents_with_scores) >= MIN_DOCUMENTS_FOR_ANSWER
        }
    
    avg_score = sum(scores) / len(scores)
    high_confidence_docs = sum(1 for score in scores if score >= threshold)
    meets_threshold = high_confidence_docs >= MIN_DOCUMENTS_FOR_ANSWER
    
    # Overall confidence is based on average score and number of high-confidence docs
    overall_confidence = min(avg_score, (high_confidence_docs / len(scores)))
    
    return {
        "overall_confidence": overall_confidence,
        "high_confidence_docs": high_confidence_docs,
        "avg_score": avg_score,
        "meets_threshold": meets_threshold,
        "total_docs": len(documents_with_scores),
        "scores": scores
    }

def filter_high_confidence_documents(documents_with_scores: List[Tuple], threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> List[Tuple]:
    """Filter documents to only include those above the confidence threshold."""
    filtered_docs = []
    
    for item in documents_with_scores:
        if isinstance(item, tuple) and len(item) == 2:
            doc, score = item
            if score is not None and score >= threshold:
                filtered_docs.append((doc, score))
        else:
            # If no score available, include the document (assume it passed some threshold)
            filtered_docs.append(item)
    
    return filtered_docs

def format_sources(documents_with_scores: List[Tuple]) -> str:
    """Format source URLs from documents into a reference string."""
    sources = set()
    
    for item in documents_with_scores:
        if isinstance(item, tuple) and len(item) == 2:
            doc, score = item
        else:
            doc = item
            score = None
        
        if hasattr(doc, 'metadata') and doc.metadata:
            source = doc.metadata.get("source", "Unknown source")
            if source and source != "unknown" and source != "Unknown source":
                sources.add(source)
    
    if sources:
        # Format sources as a numbered list
        source_list = list(sources)
        formatted_sources = "\n".join([f"{i+1}. {source}" for i, source in enumerate(source_list)])
        return f"\n\n**Sources:**\n{formatted_sources}"
    else:
        return "\n\n**Source:** Information from 2wrap.com website"

def build_enhanced_context(documents_with_scores: List[Tuple]) -> str:
    """Build enhanced context string with document metadata and confidence information."""
    if not documents_with_scores:
        return "No relevant information found in the website content."
    
    context_parts = []
    
    for i, item in enumerate(documents_with_scores):
        if isinstance(item, tuple) and len(item) == 2:
            doc, score = item
        else:
            doc = item
            score = None
        
        # Extract metadata information
        source = doc.metadata.get("source", "Unknown source") if hasattr(doc, 'metadata') else "Unknown source"
        content_type = doc.metadata.get("content_type", "general") if hasattr(doc, 'metadata') else "general"
        service_types = doc.metadata.get("service_types", []) if hasattr(doc, 'metadata') else []
        price_info = doc.metadata.get("price_info") if hasattr(doc, 'metadata') else None
        
        # Format score information
        score_str = f" (confidence: {score:.3f})" if score is not None else ""
        
        # Format service types
        service_str = f" [Services: {', '.join(service_types)}]" if service_types else ""
        
        # Format price info
        price_str = f" [Pricing: {price_info}]" if price_info else ""
        
        # Build document header
        doc_header = f"[Document {i+1}] {content_type.title()} from {source}{score_str}{service_str}{price_str}"
        
        # Add the document content
        context_parts.append(f"{doc_header}\n{doc.page_content}")
    
    return "\n\n".join(context_parts)

def generate_answer(query, retriever, confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
    """
    Generate an answer to a query using a retriever and LLM with confidence scoring
    
    Args:
        query: The user's question
        retriever: A retriever object that can fetch relevant documents
        confidence_threshold: Minimum confidence threshold for documents
        
    Returns:
        Generated answer text with confidence information
        
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
        results = retriever.get_relevant_documents(query)
        
        retrieval_time = time.time() - start_time
        logger.info(f"Vector search completed in {retrieval_time:.2f} seconds")
        
        # Process the results - they could be Documents or (Document, score) tuples
        docs_with_scores = []
        
        for item in results:
            # Check if the item is a tuple (Document, score)
            if isinstance(item, tuple) and len(item) == 2:
                doc, score = item
                docs_with_scores.append((doc, score))
            else:
                # It's just a Document, assume high confidence
                docs_with_scores.append((item, 0.8))
        
        logger.info(f"Retrieved {len(docs_with_scores)} documents from vector store")
        
        # Calculate confidence metrics
        confidence_metrics = calculate_confidence_score(docs_with_scores, confidence_threshold)
        logger.info(f"Confidence metrics: {confidence_metrics}")
        
        # Filter documents by confidence threshold
        high_confidence_docs = filter_high_confidence_documents(docs_with_scores, confidence_threshold)
        
        if not confidence_metrics["meets_threshold"] or not high_confidence_docs:
            logger.warning(f"No high-confidence documents found (threshold: {confidence_threshold})")
            logger.info(f"Found {len(docs_with_scores)} total documents, but only {confidence_metrics['high_confidence_docs']} meet confidence threshold")
            
            # Return a response indicating low confidence
            sources_str = format_sources(docs_with_scores)
            return f"""I don't have enough high-confidence information to answer your question accurately. 

The documents I found don't seem closely related to your query (confidence below {confidence_threshold:.1%}). Could you try rephrasing your question or asking about something more specific to 2wrap's car wrapping and detailing services?

If you're looking for information about our services, pricing, or company details, please feel free to ask a more specific question.{sources_str}

**Confidence Score:** {confidence_metrics['overall_confidence']:.1%}"""
        
        # Limit to top documents to avoid overwhelming the context
        top_docs = high_confidence_docs[:MAX_DOCUMENTS_FOR_CONTEXT]
        
        # Log documents and scores
        for i, (doc, score) in enumerate(top_docs):
            source = doc.metadata.get("source", "Unknown source") if hasattr(doc, 'metadata') else "Unknown source"
            logger.info(f"High-confidence Document {i+1} from {source} with score: {score:.4f}")
            
            # Log a preview of each document
            preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
            logger.info(f"Document {i+1} preview: {preview}")
        
        # Build enhanced context
        context = build_enhanced_context(top_docs)
        
        # Step 2: Generate answer using LLM with enhanced context
        logger.info("RAG Step 2: Generating answer using LLM with enhanced context")
        
        prompt = f"""You are an AI assistant for 2wrap.com, a car wrapping and detailing company. You must answer the user's question based ONLY on the provided context from the 2wrap website.

CONTEXT FROM 2WRAP WEBSITE:
{context}

USER QUESTION:
{query}

STRICT INSTRUCTIONS:
1. Answer ONLY based on the information provided in the context above
2. If the context doesn't contain enough information to answer fully, clearly state what information is missing
3. Always mention that the information comes from 2wrap.com
4. Include specific details from the context when relevant (prices, services, processes, etc.)
5. If discussing services, mention what 2wrap specifically offers based on the context
6. Be conversational and helpful, but stay strictly within the provided information
7. DO NOT use general knowledge about car wrapping or detailing that isn't in the context
8. If you cannot answer based on the context, say so clearly and suggest contacting 2wrap directly

FORMAT YOUR ANSWER:
- Provide a direct, helpful answer based on the context
- Reference specific information from 2wrap when relevant
- Be clear about what 2wrap offers vs. general industry practices
- Keep the tone professional but friendly

YOUR ANSWER:"""
        
        logger.info("Sending enhanced prompt to LLM")
        start_time = time.time()
        response = get_gemini_response(prompt)
        generation_time = time.time() - start_time
        
        logger.info(f"LLM response generated in {generation_time:.2f} seconds")
        
        # Add source references and confidence information
        sources_str = format_sources(top_docs)
        confidence_str = f"\n\n**Confidence Score:** {confidence_metrics['overall_confidence']:.1%} (based on {confidence_metrics['high_confidence_docs']} high-confidence sources)"
        
        final_answer = f"{response}{sources_str}{confidence_str}"
        
        logger.info("RAG process completed successfully")
        
        return final_answer
        
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}")
        raise

def generate_answer_with_custom_threshold(query, confidence_threshold: float = 0.6):
    """
    Generate an answer with a custom confidence threshold using the score-threshold retriever.
    
    Args:
        query: The user's question
        confidence_threshold: Custom confidence threshold (0.0 to 1.0)
        
    Returns:
        Generated answer text with confidence information
    """
    try:
        # Get retriever with custom score threshold
        retriever = get_vector_store_with_score_threshold(
            score_threshold=confidence_threshold,
            k=MAX_DOCUMENTS_FOR_CONTEXT
        )
        
        return generate_answer(query, retriever, confidence_threshold)
        
    except Exception as e:
        logger.error(f"Error generating answer with custom threshold: {str(e)}")
        raise
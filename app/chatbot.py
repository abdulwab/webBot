from app.llm import get_gemini_response

def generate_answer(query, retriever):
    docs = retriever.get_relevant_documents(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = f"""You are an assistant for an e-commerce store.
Use the context below to answer the user's question.

Context:
{context}

Question:
{query}

Answer:"""
    return get_gemini_response(prompt)
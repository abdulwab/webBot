# Web Chatbot with RAG

A FastAPI-based chatbot that uses Retrieval-Augmented Generation (RAG) to answer questions about web content. The application scrapes a given URL, chunks the content, creates embeddings, stores them in Pinecone, and then uses the Gemini API to generate answers based on the retrieved context.

## Features

- Web scraping with BeautifulSoup
- Text chunking with LangChain
- Vector embeddings with OpenAI
- Vector storage with Pinecone
- LLM integration with Google's Gemini API
- FastAPI endpoints with Swagger UI documentation

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_ENVIRONMENT=your_pinecone_environment
PINECONE_INDEX_NAME=your_pinecone_index_name
GOOGLE_API_KEY=your_google_api_key
OPENAI_API_KEY=your_openai_api_key
```

## API Endpoints

### GET /

A simple health check endpoint that returns a status message.

### POST /init

Initialize the chatbot by scraping a URL, chunking the content, and creating a vector store.

**Request Body:**
```json
{
  "url": "https://example.com"
}
```

**Response:**
```json
{
  "status": "initialized",
  "chunks": 10
}
```

### POST /chat

Ask a question about the scraped content.

**Request Body:**
```json
{
  "query": "What is this website about?"
}
```

**Response:**
```json
{
  "answer": "This website is about..."
}
```

## Running Locally

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the server:
   ```
   uvicorn app.main:app --reload
   ```

3. Visit `http://localhost:8000/docs` to access the Swagger UI documentation.

## Deployment

This application is configured for deployment on Railway. Simply push to your repository and Railway will handle the deployment process.

## Error Handling

The application includes comprehensive error handling and logging for all components, including:
- URL validation and scraping errors
- Text chunking validation
- Pinecone connection and index management
- OpenAI API integration
- Gemini API integration

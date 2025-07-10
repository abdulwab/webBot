import os
import pinecone
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

load_dotenv()

# Load from .env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")


# Initialize Pinecone client
def init_pinecone():
    if not pinecone.list_indexes():
        pinecone.init(
            api_key=PINECONE_API_KEY,
            environment=PINECONE_ENVIRONMENT
        )


# Create embedding model
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Create a new vector store and upsert documents into Pinecone
def create_vector_store(documents: list[Document]):
    init_pinecone()
    embedder = get_embedding_model()

    if PINECONE_INDEX_NAME not in pinecone.list_indexes():
        pinecone.create_index(PINECONE_INDEX_NAME, dimension=384, metric="cosine")

    return LangchainPinecone.from_documents(
        documents=documents,
        embedding=embedder,
        index_name=PINECONE_INDEX_NAME
    )


# Load existing vector store and return retriever
def get_vector_store_retriever(k: int = 3):
    init_pinecone()
    embedder = get_embedding_model()

    return LangchainPinecone.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embedder
    ).as_retriever(search_kwargs={"k": k})
import os
from dotenv import load_dotenv
from langchain.vectorstores import Pinecone as LangchainPinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from pinecone import Pinecone

load_dotenv()

# Load from .env
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)


def get_embedding_model():
    """Create embedding model."""
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def create_vector_store(documents: list[Document]):
    """Create a new vector store and upsert documents into Pinecone."""
    embedder = get_embedding_model()

    if PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(name=PINECONE_INDEX_NAME, dimension=384, metric="cosine")

    return LangchainPinecone.from_documents(
        documents=documents,
        embedding=embedder,
        index_name=PINECONE_INDEX_NAME
    )


def get_vector_store_retriever(k: int = 3):
    """Load existing vector store and return retriever."""
    embedder = get_embedding_model()

    return LangchainPinecone.from_existing_index(
        index_name=PINECONE_INDEX_NAME,
        embedding=embedder
    ).as_retriever(search_kwargs={"k": k})
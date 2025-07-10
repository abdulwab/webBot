from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings

def get_embedding_model():
    return HuggingFaceInferenceAPIEmbeddings(
        api_key=None,  # You can skip or use HF token
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
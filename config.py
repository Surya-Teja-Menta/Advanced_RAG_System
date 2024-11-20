import os

# Constants
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 300
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-l6-v2"

# Path Configurations
BASE_STORAGE_PATH = os.path.join(os.getcwd(), "storage")
CHROMA_PATH = os.path.join(BASE_STORAGE_PATH, "chroma")
BM25_PATH = os.path.join(BASE_STORAGE_PATH, "bm25")

# Retrieval Configuration
DEFAULT_VECTOR_K = 6
DEFAULT_BM25_K = 4
DEFAULT_SUB_QUERIES = 4
DEFAULT_VIEWPOINTS = 4
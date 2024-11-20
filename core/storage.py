import os
import shutil

class PersistentStorage:
    def __init__(self, chroma_path: str, bm25_path: str):
        """
        Initialize persistent storage for document embeddings.
        
        Args:
            chroma_path (str): Path for Chroma vector store
            bm25_path (str): Path for BM25 retriever storage
        """
        self.chroma_path = chroma_path
        self.bm25_path = bm25_path
        self.initialize_storage()

    def initialize_storage(self):
        """Create storage directories if they don't exist"""
        os.makedirs(self.chroma_path, exist_ok=True)
        os.makedirs(self.bm25_path, exist_ok=True)

    def clear_storage(self):
        """Clear existing storage directories"""
        for path in [self.chroma_path, self.bm25_path]:
            if os.path.exists(path):
                shutil.rmtree(path)
                os.makedirs(path)

    def storage_exists(self) -> bool:
        """
        Check if embeddings storage exists and contains data.
        
        Returns:
            bool: True if storage exists and has content, False otherwise
        """
        return (os.path.exists(self.chroma_path) and 
                os.path.exists(self.bm25_path) and 
                any(os.scandir(self.chroma_path)))
from dataclasses import dataclass
from typing import Optional

@dataclass
class RetrievalConfig:
    """Configuration for document retrieval strategies"""
    vector_k: int = 6
    bm25_k: int = 4
    sub_queries: int = 4
    viewpoints: int = 4
    chunk_size: int = 1500
    chunk_overlap: int = 300
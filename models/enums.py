from enum import Enum, auto
from typing import List, Dict, Any

class QueryType(Enum):
    """
    Enumeration of different query types for classification
    """
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    OPINION = "opinion"
    CONTEXTUAL = "contextual"
    GREETING = "greeting"

    @classmethod
    def get_processing_instructions(cls) -> Dict[str, str]:
        """
        Provides processing instructions for each query type
        """
        return {
            cls.FACTUAL: "Provide precise, verifiable information",
            cls.ANALYTICAL: "Break down the analysis step by step",
            cls.OPINION: "Present different viewpoints and perspectives",
            cls.CONTEXTUAL: "Consider the user's specific context",
            cls.GREETING: "Respond warmly and professionally"
        }
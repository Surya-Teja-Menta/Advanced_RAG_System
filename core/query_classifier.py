from models.enums import QueryType

class QueryClassifier:
    def __init__(self, llm):
        """
        Initialize query classifier with a language model.
        
        Args:
            llm: Language model for query classification
        """
        self.llm = llm
        self.classification_prompt = """
        Classify the following query into one of these categories: 
        - FACTUAL (specific, verifiable information)
        - ANALYTICAL (comprehensive analysis)
        - OPINION (subjective matters)
        - CONTEXTUAL (user-specific context)
        - GREETING (general greetings or pleasantries)
        
        Output only one word from the above categories.
        
        Query: {query}
        Classification:"""
    
    async def classify(self, query: str) -> QueryType:
        """
        Classify the query into a QueryType.
        
        Args:
            query (str): User's input query
        
        Returns:
            QueryType: Classified query type
        """
        response = await self.llm.ainvoke(self.classification_prompt.format(query=query))
        
        # Handle response extraction
        classification = response.content if hasattr(response, 'content') else str(response)
        classification = classification.upper().strip()
        
        # Normalize classification
        classification_mapping = {
            'FACT': 'FACTUAL',
            'FACTS': 'FACTUAL',
            'ANALYZE': 'ANALYTICAL',
            'ANALYSIS': 'ANALYTICAL',
            'OPINIONS': 'OPINION',
            'OPINIONATED': 'OPINION',
            'CONTEXT': 'CONTEXTUAL',
            'GREET': 'GREETING',
            'GREETINGS': 'GREETING',
            'HELLO': 'GREETING',
            'HI': 'GREETING'
        }
        
        # Extract first word and map to QueryType
        classification = classification.split()[0] if classification else 'CONTEXTUAL'
        classification = classification_mapping.get(classification, classification)
        
        try:
            return QueryType[classification]
        except KeyError:
            return QueryType.CONTEXTUAL
from typing import Dict, Any
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from models.enums import QueryType

class EnhancedRAGChain:
    def __init__(self, retriever, classifier, greeting_handler, memory, llm):
        """
        Initialize the Enhanced RAG Chain.
        
        Args:
            retriever: Document retrieval system
            classifier: Query type classifier
            greeting_handler: Handles greeting interactions
            memory: Conversation memory
            llm: Language model
        """
        self.retriever = retriever
        self.classifier = classifier
        self.greeting_handler = greeting_handler
        self.memory = memory
        self.llm = llm
        self.prompt_template = self._create_prompt_template()

    def _create_prompt_template(self):
        """Create prompt template for RAG processing"""
        system_template = """You are an interactive research Analyst that helps users by finding and sharing relevant information. 
        Use the following pieces of context to answer the user's question.
        If you don't know the answer, just say that you don't know, don't try to make up an answer. 
        Cross check the answer with query/question before confirming.
        
        Query Type: {query_type}
        Context: {context}
        Question: {question}
        
        Additional Instructions based on query type:
        - For Factual queries: Provide precise, verifiable information
        - For Analytical queries: Break down the analysis step by step
        - For Opinion queries: Present different viewpoints and perspectives
        - For Contextual queries: Consider the user's specific context
        - For Greetings: Respond warmly and professionally
        
        ALWAYS include SOURCES in your answer.
        """
        return ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}")
        ])

    async def process_query(self, query: str, context: str = None) -> Dict[str, Any]:
        """
        Process user query through RAG pipeline.
        
        Args:
            query (str): User's input query
            context (str, optional): Conversation context
        
        Returns:
            Dict[str, Any]: Processing result with answer, sources, and query type
        """
        query_type = await self.classifier.classify(query)
        
        # Handle greeting separately
        if query_type == QueryType.GREETING:
            greeting_response = await self.greeting_handler.handle_greeting(query)
            return {
                "answer": greeting_response,
                "sources": [],
                "query_type": query_type
            }
        
        # Retrieve documents
        docs = await self.retriever.retrieve_documents(query, query_type, context)
        
        # Generate response
        context_text = "\n".join(doc.page_content for doc in docs)
        response = await self.llm.ainvoke(
            self.prompt_template.format(
                query_type=query_type.value,
                context=context_text,
                question=query
            )
        )

        return {
            "answer": response,
            "sources": docs,
            "query_type": query_type
        }
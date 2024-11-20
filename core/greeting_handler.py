class GreetingHandler:
    def __init__(self, llm):
        """
        Initialize greeting handler with a language model.
        
        Args:
            llm: Language model for generating responses
        """
        self.llm = llm
        self.greeting_prompt = """
        You are a friendly and engaging assistant. Respond naturally and appropriately to this casual interaction:
        
        User's message: {message}
        
        Remember to:
        - Be natural and conversational
        - Match the tone of the user
        - If they're sharing something personal, show appropriate empathy
        - If they ask for a joke, be appropriately humorous
        - If they ask how you're doing, be positive but honest about being an AI
        - Keep the response concise but engaging
        
        Response:"""

    async def handle_greeting(self, message: str) -> str:
        """
        Generate a contextual response to a greeting or casual message.
        
        Args:
            message (str): User's input message
        
        Returns:
            str: Generated conversational response
        """
        response = await self.llm.ainvoke(self.greeting_prompt.format(message=message))
        return response.content if hasattr(response, 'content') else str(response)
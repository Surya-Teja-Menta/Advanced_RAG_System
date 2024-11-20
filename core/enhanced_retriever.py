from typing import List, Optional
from langchain.docstore.document import Document
from models.enums import QueryType
from models.configs import RetrievalConfig

class EnhancedRetriever:
    def __init__(self, vector_retriever, bm25_retriever, llm, config: RetrievalConfig):
        """
        Initialize advanced document retriever.
        
        Args:
            vector_retriever: Vector-based document retriever
            bm25_retriever: BM25 document retriever
            llm: Language model for query enhancement
            config: Retrieval configuration
        """
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.llm = llm
        self.config = config

    async def enhance_query(self, query: str) -> str:
        """Enhance query for better retrieval"""
        prompt = f"""Enhance this query for better information retrieval. 
        Return a single enhanced query, not a list. 
        Query: {query}"""
        response = await self.llm.ainvoke(prompt)
        return response.content.split('\n')[0] if '\n' in response.content else response.content

    async def generate_sub_queries(self, query: str) -> List[str]:
        """Generate sub-queries for comprehensive retrieval"""
        prompt = f"Generate {self.config.sub_queries} specific sub-questions for: {query}"
        response = await self.llm.ainvoke(prompt)
        return response.content.split('\n')

    async def identify_viewpoints(self, query: str) -> List[str]:
        """Identify diverse viewpoints for the query"""
        prompt = f"Identify {self.config.viewpoints} distinct viewpoints on: {query}"
        response = await self.llm.ainvoke(prompt)
        return response.content.split('\n')

    async def retrieve_documents(self, 
                                 query: str, 
                                 query_type: QueryType, 
                                 context: Optional[str] = None) -> List[Document]:
        """
        Retrieve documents based on query type and context.
        
        Args:
            query (str): User's input query
            query_type (QueryType): Classified query type
            context (Optional[str]): Additional context
        
        Returns:
            List[Document]: Retrieved and processed documents
        """
        enhanced_query = await self.enhance_query(query)
        
        if query_type == QueryType.FACTUAL:
            vector_docs = await self.vector_retriever.aget_relevant_documents(enhanced_query)
            bm25_docs = await self.bm25_retriever.aget_relevant_documents(enhanced_query)
            return self._merge_and_deduplicate(vector_docs, bm25_docs)
        
        elif query_type == QueryType.ANALYTICAL:
            sub_queries = await self.generate_sub_queries(query)
            all_docs = []
            for sub_query in sub_queries:
                docs = await self.vector_retriever.aget_relevant_documents(sub_query)
                all_docs.extend(docs)
            return self._ensure_diversity(all_docs)
        
        elif query_type == QueryType.OPINION:
            viewpoints = await self.identify_viewpoints(query)
            all_docs = []
            for viewpoint in viewpoints:
                combined_query = f"{query} {viewpoint}"
                docs = await self.vector_retriever.aget_relevant_documents(combined_query)
                all_docs.extend(docs)
            return self._ensure_diversity(all_docs)
        
        elif query_type == QueryType.CONTEXTUAL and context:
            contextualized_query = f"Given context: {context}, {query}"
            vector_docs = await self.vector_retriever.aget_relevant_documents(contextualized_query)
            bm25_docs = await self.bm25_retriever.aget_relevant_documents(contextualized_query)
            return self._merge_and_deduplicate(vector_docs, bm25_docs)
        
        else:
            return await self.vector_retriever.aget_relevant_documents(enhanced_query)

    def _merge_and_deduplicate(self, docs1: List[Document], docs2: List[Document]) -> List[Document]:
        """Merge and deduplicate documents"""
        seen = set()
        merged = []
        for doc in docs1 + docs2:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                merged.append(doc)
        return merged[:self.config.vector_k]

    def _ensure_diversity(self, docs: List[Document]) -> List[Document]:
        """Ensure document diversity"""
        return docs[:self.config.vector_k]
from langchain.docstore.document import Document
import pymupdf4llm
from langchain.text_splitter import RecursiveCharacterTextSplitter

class DocumentLoader:
    @staticmethod
    async def load_pdf_documents(file_paths, chunk_size=1500, chunk_overlap=300):
        """
        Load PDF documents and split them into chunks
        
        Args:
            file_paths (list): List of file paths to PDF documents
            chunk_size (int): Size of text chunks
            chunk_overlap (int): Overlap between chunks
        
        Returns:
            list: List of Document objects
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        documents = []
        for file_path in file_paths:
            # Convert PDF to markdown
            pdf_doc = pymupdf4llm.to_markdown(file_path)
            
            # Split text into chunks
            texts = text_splitter.split_text(pdf_doc)
            
            # Convert texts to Document objects
            docs = [
                Document(
                    page_content=text, 
                    metadata={"source": file_path}
                ) for text in texts
            ]
            
            documents.extend(docs)
        
        return documents
import os
import logging
from typing import List
import hashlib
from datetime import datetime

import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration constants
PDFS_DIR = "pdfs/"
MODEL_NAME = "deepseek-r1:1.5b"

# Ensure PDF storing dir exists
os.makedirs(PDFS_DIR, exist_ok=True)

def calculate_file_hash(file_content: bytes) -> str:
    """Calculate SHA-256 hash of file content."""
    return hashlib.sha256(file_content).hexdigest()


class DocumentProcessor:
    """Handles document processing operations including loading, splitting, and indexing."""
    def __init__(self, model_name: str):
        self.embeddings = OllamaEmbeddings(model=model_name)
        self.vector_store = None
        self.current_file_hash = None

    def process_file(self, file) -> bool:
        """Process file and return whether reprocessing was needed."""
        try:
            file_content = file.getvalue()
            file_hash = calculate_file_hash(file_content)

            # Check if file has already been processed
            if file_hash == self.current_file_hash and self.vector_store is not None:
                logger.info("File already processed, skipping reprocessing")
                return False
            
            # Process new file
            file_path = self.upload_pdf(file)
            documents = self.load_pdf(file_path)
            chunked_documents = self.split_text(documents)
            self.index_documents(chunked_documents)
            self.current_file_hash = file_hash
            return True
        
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            raise

    def upload_pdf(self, file) -> str:
        """Upload PDF file to temporary directory."""
        try:
            file_path = os.path.join(PDFS_DIR, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getvalue())
            return file_path
        except Exception as e:
            logger.error(f"Error uploading PDF: {str(e)}")
            raise
    
    def load_pdf(self, file_path: str) -> List[Document]:
        """Load PDF file and return documents."""
        try:
            loader = PDFPlumberLoader(file_path)
            documents = loader.load()
            # Clean up temporary file after loading
            os.remove(file_path)
            return documents
        except Exception as e:
            logger.error(f"Error loading PDF: {str(e)}")
            raise
    
    def split_text(self, documents: List[Document]) -> List[Document]:
        """Split documents into semantic chunks."""
        try:
            text_splitter = SemanticChunker(self.embeddings)
            return text_splitter.split_documents(documents)
        except Exception as e:
            logger.error(f"Error splitting documents: {str(e)}")
            raise
    
    def index_documents(self, documents: List[Document]) -> None:
        """Create new in-memory FAISS index with document embeddings."""
        try:
            self.vector_store = FAISS.from_documents(documents, self.embeddings)
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            raise
    
    def retrieve_relevant_docs(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve relevant documents for the given query."""
        try:
            if not self.vector_store:
                raise ValueError("No documents have been indexed yet.")
            return self.vector_store.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}") 
            raise


class QuestionAnswerer:
    """Handles question answering using retrieved documents."""

    def __init__(self, model_name: str):
        self.model = OllamaLLM(model=model_name)
        self.prompt = ChatPromptTemplate.from_template("""
            You are an assistant for question-answering tasks. 
            Use the following pieces of retrieved context to answer the question. 
            If you don't know the answer, just say that you don't know but don't make up an answer on your own. 

            Use 3 to 4 sentences maximum and keep the answer concise.
            Question: {question} 

            Context: {context} 

            Answer: 
        """)

    def answer_question(self, question: str, documents: List[Document]) -> str:
        """Generate answer based on question and relevant documents."""
        try:
            context = "\n\n".join([doc.page_content for doc in documents])
            chain = self.prompt | self.model
            return chain.invoke({"question": question, "context": context})
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}")
            raise


def initialize_session_state():
    """Initialize Streamlit session state variables."""
    if "doc_processor" not in st.session_state:
        st.session_state.doc_processor = DocumentProcessor(MODEL_NAME)
    if "qa_system" not in st.session_state:
        st.session_state.qa_system = QuestionAnswerer(MODEL_NAME)
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []



def main():
    """Main application function."""
    st.title("PDF Q&A")

    initialize_session_state()

    # File upload
    uploaded_file = st.file_uploader(
        "Upload PDF",
        type="pdf",
        accept_multiple_files=False,
        key="pdf_uploader"
    )


    # Aftter file upload
    if uploaded_file:
        try:
            # Process uploaded file if needed
            if st.session_state.doc_processor.process_file(uploaded_file):
                st.success("New document processed and indexed successfully!")
            
            # Display chat history
            for chat in st.session_state.chat_history:
                st.chat_message("user").write(chat['question'])
                st.chat_message("assistant").write(chat['answer'])

            # Chat interface
            question = st.chat_input("Ask a question about your document")

            if question:
                
                # Get and display answer
                related_documents = st.session_state.doc_processor.retrieve_relevant_docs(question)
                answer = st.session_state.qa_system.answer_question(question, related_documents)

                # Display current question-answer
                st.chat_message("user").write(question)
                st.chat_message("assistant").write(answer)

                # Save to chat history
                st.session_state.chat_history.append({"question": question, "answer": answer})

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logger.error(f"Application error: {str(e)}")


if __name__ == "__main__":
    main()










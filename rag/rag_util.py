# rag/rag_utils.py
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.chains import RetrievalQA
from langchain_together import Together
import streamlit as st

def initialize_rag(fixed_role_dialogue: str, together_api_key: str):
    """Initialize RAG system with fixed_role_dialogue."""
    try:
        if not fixed_role_dialogue or not isinstance(fixed_role_dialogue, str):
            st.error("Invalid or empty fixed_role_dialogue.")
            return None, None, None
        
        # Create document
        documents = [Document(page_content=fixed_role_dialogue)]
        
        # Split text
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        texts = text_splitter.split_documents(documents)
        
        # Create embeddings and vector store
        embedding_model_name = "emilyalsentzer/Bio_ClinicalBERT"
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        vectorstore = FAISS.from_documents(texts, embeddings)
        retriever = vectorstore.as_retriever()
        
        # Initialize LLM
        llm = Together(
            model="google/gemma-2-27b-it",
            together_api_key="38a11d9280e22f5b8c2e38385f133672f06cd405ca1f2cbfd7216183c451a33e",
            temperature=0.7,
            max_tokens=300
        )
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        return qa_chain, fixed_role_dialogue, retriever
    except Exception as e:
        st.error(f"Error initializing RAG: {e}")
        return None, None, None
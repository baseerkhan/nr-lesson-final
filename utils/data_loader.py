"""
data_loader.py - Utilities for loading and generating sample data

© 2025 NextRun Digital. All Rights Reserved.
"""

import os
import json
import pandas as pd
from pathlib import Path
from .config import DATA_DIR
from .embedding import EmbeddingManager

def create_sample_documents():
    """Create sample documents for the knowledge base"""
    
    from .config import get_openai_api_key
    
    # Check if OpenAI API key is configured
    if not get_openai_api_key():
        return 0
    
    # Sample data based on the course slides and legal domain context
    sample_docs = [
        {
            "title": "Augmented LLM Overview", 
            "text": "Augmented LLM systems have four key characteristics: 1. Memory - letting the LLM remember past interactions, 2. Information Retrieval - adding RAG for context, 3. Tool Usage - giving the LLM access to functions and APIs, 4. Workflow Control - the LLM output controls which tools are used."
        },
        {
            "title": "Memory and Context Engineering", 
            "text": "Memory systems retain past interactions so the LLM can continue intelligently across sessions. Context engineering supplies the right information at the right time to enhance model output. Historical memory without RAG has simple implementation but limited context scope. With RAG, the system dynamically fetches fresh context and improves grounding."
        },
        {
            "title": "Retrieval Augmented Generation", 
            "text": "RAG is a popular technique that enhances LLM responses by retrieving relevant external knowledge from a knowledge base before generating an answer. It improves accuracy, reduces hallucinations, and allows the model to provide contextually relevant and up-to-date information."
        },
        {
            "title": "Embeddings and Vector Databases", 
            "text": "Embeddings are numerical representations that capture meaning. They're created through models like Word2Vec and BERT, using cosine similarity for comparison. Vector databases store these embeddings for efficient similarity search, which powers applications like RAG, semantic search, and recommendation engines."
        },
        {
            "title": "Tool and Function Calling", 
            "text": "Tool or function calling lets an LLM call external tools or APIs to fetch live data, do calculations, access databases, and trigger business processes. The LLM generates a special output describing which tool to call and what parameters to send, then the framework runs the tool and feeds results back to the LLM."
        },
        {
            "title": "Model Context Protocol", 
            "text": "MCP (Model Context Protocol) is a standard way for LLMs to discover and call tools. It defines tool names, input parameters, and expected output formats. MCP matters because it avoids ad hoc formats, makes LLMs interoperable across different apps, helps manage security and validation, and enables complex workflows through agent orchestration."
        },
        {
            "title": "SmartAdvocate Integration Naming Conventions", 
            "text": "The SmartAdvocate integration uses a standardized file naming convention of [CaseID]_[DocType]_[VersionDate].ext for document uploads. This convention ensures proper document tracking and version control in the legal case management system. For example, 'ABC123_Medical_20250601.pdf' represents medical records for case ABC123 dated June 1, 2025."
        },
        {
            "title": "SmartAdvocate API Configuration", 
            "text": "Configured API polling and document upload protocols for SmartAdvocate integration handle automatic synchronization between AI systems and the case management platform. The system polls for updates every 15 minutes and automatically categorizes incoming documents based on embedded metadata."
        },
        {
            "title": "Law Firm AI Agent Framework", 
            "text": "Implemented AI agents for law firm automation include specialized tools like Medical Record Chase, Document Classification, Bill of Particulars Generation, and SmartAdvocate integration. These agents work together in an orchestrated workflow to minimize manual document handling and accelerate case processing."
        },
        {
            "title": "Document Classification for Legal Documents", 
            "text": "The Document Classification agent uses embeddings to categorize incoming legal documents by type, such as medical records, court filings, or client correspondence. This allows for automated routing and filing within case management systems. The classification model achieves 94% accuracy across 27 document categories."
        },
        {
            "title": "Medical Record Chase Automation", 
            "text": "The Medical Record Chase agent automates the process of requesting and following up on medical records from healthcare providers, tracking receipt status, and identifying missing documents critical to personal injury cases. The system generates customized follow-up schedules based on provider response patterns."
        },
        {
            "title": "Bill of Particulars Generation", 
            "text": "The Bill of Particulars Generation agent extracts relevant information from medical records and case notes to automatically draft Bills of Particulars documents. The system uses RAG to find similar past cases and adapts their language to the current case facts, reducing drafting time by 75%."
        }
    ]
    
    # Create embedding manager
    embedding_manager = EmbeddingManager()
    
    # Add each document to knowledge base
    for doc in sample_docs:
        embedding_manager.add_document(
            text=doc["text"],
            metadata={"title": doc["title"], "source": "course_material"}
        )
    
    return len(sample_docs)

def ensure_data_directory():
    """Ensure data directory exists"""
    os.makedirs(DATA_DIR, exist_ok=True)
    
def initialize_data(force_reload=False):
    """Initialize data for the application"""
    ensure_data_directory()
    
    # Create knowledge base file if it doesn't exist or force reload is requested
    kb_path = DATA_DIR / "knowledge_base.xlsx"
    
    if not os.path.exists(kb_path) or force_reload:
        # If force reload and file exists, delete the old file
        if force_reload and os.path.exists(kb_path):
            print("Force reloading knowledge base - deleting existing file")
            os.remove(kb_path)
            
        print("Creating sample documents for knowledge base...")
        docs_created = create_sample_documents()
        print(f"Created {docs_created} sample documents.")
        
        # Verify documents were created successfully
        if os.path.exists(kb_path):
            try:
                df = pd.read_excel(kb_path)
                print(f"Verification: Knowledge base contains {len(df)} documents")
                return len(df) > 0
            except Exception as e:
                print(f"Error verifying knowledge base: {e}")
                return False
        return docs_created > 0
    else:
        # Verify existing knowledge base
        try:
            df = pd.read_excel(kb_path)
            print(f"Using existing knowledge base with {len(df)} documents")
            return len(df) > 0
        except Exception as e:
            print(f"Error reading existing knowledge base: {e}")
            return False

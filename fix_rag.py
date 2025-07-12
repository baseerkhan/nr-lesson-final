"""
fix_rag.py - Fixes the RAG Demo by resetting and verifying the knowledge base

© 2025 NextRun Digital. All Rights Reserved.
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path

# Add the project directory to Python path
sys.path.append(str(Path(__file__).parent))

# Import utilities
from utils.config import DATA_DIR, get_openai_api_key
from utils.embedding import EmbeddingManager
from utils.rag import RAGSystem
from utils.data_loader import create_sample_documents

def fix_knowledge_base():
    """Fix the knowledge base by deleting it and creating a new one"""
    print("Fixing RAG Demo knowledge base...")
    
    # Check OpenAI API key
    api_key = get_openai_api_key()
    if not api_key:
        print("OpenAI API key not found! Please set the OPENAI_API_KEY environment variable.")
        return False
        
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Remove existing knowledge base if it exists
    kb_path = DATA_DIR / "knowledge_base.xlsx"
    if os.path.exists(kb_path):
        print(f"Removing existing knowledge base at {kb_path}")
        os.remove(kb_path)
    
    # Create sample documents
    print("Creating new knowledge base with sample documents...")
    docs_created = create_sample_documents()
    print(f"Created {docs_created} sample documents")
    
    # Verify the knowledge base was created
    if not os.path.exists(kb_path):
        print("ERROR: Knowledge base file was not created!")
        return False
        
    # Check if the file contains data
    try:
        df = pd.read_excel(kb_path)
        print(f"Knowledge base contains {len(df)} documents")
        if len(df) == 0:
            print("ERROR: Knowledge base is empty!")
            return False
    except Exception as e:
        print(f"ERROR: Failed to read knowledge base: {e}")
        return False
    
    return True

def test_rag():
    """Test the RAG system with sample queries"""
    # Initialize RAG components
    embedding_manager = EmbeddingManager()
    rag_system = RAGSystem(embedding_manager)
    
    # Sample queries to test
    test_queries = [
        "What are the key concepts of augmented LLMs?",
        "Tell me about law firm automation using AI agents",
        "What triggered the AI Frenzy?",
        "What is RAG and how does it work?"
    ]
    
    print("\nTesting RAG with sample queries:")
    for query in test_queries:
        print(f"\n>>> Query: {query}")
        
        # Execute query
        result = rag_system.query(query)
        
        # Display result
        print(f"\nAnswer: {result['answer']}")
        
        # Check if it's the default "I don't have sufficient information" response
        if "I don't have sufficient information" in result['answer']:
            print("WARNING: Got default 'insufficient information' response")
        
    print("\nRAG testing complete!")

if __name__ == "__main__":
    # Fix knowledge base
    if fix_knowledge_base():
        # Test RAG
        test_rag()
    else:
        print("Failed to fix knowledge base. Cannot proceed with testing.")

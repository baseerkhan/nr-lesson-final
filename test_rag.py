"""
test_rag.py - Test script for RAG functionality

© 2025 NextRun Digital. All Rights Reserved.
"""

import os
import sys
from pathlib import Path
import json

# Add the project directory to Python path
sys.path.append(str(Path(__file__).parent))

from utils.config import DATA_DIR, get_openai_api_key
from utils.embedding import EmbeddingManager
from utils.rag import RAGSystem
from utils.data_loader import initialize_data, create_sample_documents

def test_knowledge_base():
    """Test if knowledge base exists and has content"""
    kb_path = DATA_DIR / "knowledge_base.xlsx"
    print(f"Knowledge base path: {kb_path}")
    print(f"Knowledge base exists: {os.path.exists(kb_path)}")
    
    embedding_manager = EmbeddingManager()
    docs = embedding_manager.get_all_documents()
    print(f"Number of documents in knowledge base: {len(docs)}")
    
    if not docs.empty:
        print("\nDocument IDs and Titles:")
        for _, row in docs.iterrows():
            metadata = json.loads(row['metadata']) if isinstance(row['metadata'], str) else row['metadata']
            title = metadata.get('title', 'No title')
            print(f"- ID: {row['id']} | Title: {title}")
    else:
        print("Knowledge base is empty.")
        
        # Create sample documents if empty
        print("\nCreating sample documents...")
        docs_created = create_sample_documents()
        print(f"Created {docs_created} sample documents.")
        
        # Check if documents were created
        docs = embedding_manager.get_all_documents()
        print(f"Number of documents after creation: {len(docs)}")

def test_rag_query(query="What triggered the AI Frenzy?"):
    """Test RAG query functionality"""
    print(f"\nTesting RAG query: '{query}'")
    
    # Initialize RAG system
    embedding_manager = EmbeddingManager()
    rag_system = RAGSystem(embedding_manager)
    
    # Execute query
    result = rag_system.query(query, top_k=3)
    
    # Display results
    print("\nQuery Result:")
    print(f"Answer: {result['answer']}")
    
    print("\nRetrieved Context:")
    print(result['context'])
    
    print("\nSources:")
    for i, source in enumerate(result['sources']):
        print(f"Document {i+1}:")
        print(f"- Similarity Score: {source['similarity']:.4f}")
        metadata = source.get('metadata', {})
        if metadata:
            print(f"- Title: {metadata.get('title', 'No title')}")
        print(f"- Text: {source['text'][:100]}...")
    
    print("\nSystem Prompt:")
    print(result['conversation']['system_prompt'])
    
    return result

if __name__ == "__main__":
    # Ensure we have an OpenAI API key
    api_key = get_openai_api_key()
    if not api_key:
        print("OpenAI API key not configured. Please set the OPENAI_API_KEY environment variable.")
        sys.exit(1)
    
    # Test knowledge base
    test_knowledge_base()
    
    # Test RAG queries
    test_rag_query("What triggered the AI Frenzy?")
    test_rag_query("Tell me about law firm automation using AI agents")
    test_rag_query("What are the key concepts of augmented LLMs?")

"""
embedding.py - Utilities for creating and managing embeddings

© 2025 NextRun Digital. All Rights Reserved.
"""

import os
import pandas as pd
import numpy as np
from openai import OpenAI
import streamlit as st
from pathlib import Path
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Union
from .config import DATA_DIR, get_openai_api_key

class EmbeddingManager:
    """Manage embeddings with Excel storage"""
    
    def __init__(self, embedding_file: str = "knowledge_base.xlsx"):
        """Initialize the embedding manager"""
        self.client = OpenAI(api_key=get_openai_api_key())
        self.embedding_path = DATA_DIR / embedding_file
        self._ensure_data_dir()
        
    def _ensure_data_dir(self):
        """Ensure data directory exists"""
        os.makedirs(DATA_DIR, exist_ok=True)
    
    def _create_embedding(self, text: str) -> List[float]:
        """Create embedding for a text using OpenAI API"""
        try:
            # Print debug info
            print(f"Creating embedding for text: '{text[:50]}...'")
            
            # Create embedding
            response = self.client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            
            # Check if response has data
            if not response.data:
                print("WARNING: Empty embedding response data from OpenAI")
                # Return empty vector as fallback
                return [0.0] * 1536  # Standard size for ada-002
                
            embedding = response.data[0].embedding
            print(f"Embedding created successfully, length: {len(embedding)}")
            
            return embedding
        except Exception as e:
            print(f"ERROR creating embedding: {e}")
            # Return empty vector as fallback
            return [0.0] * 1536  # Standard size for ada-002
    
    def _cosine_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate cosine similarity between two embeddings"""
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        norm1 = sum(a * a for a in embedding1) ** 0.5
        norm2 = sum(b * b for b in embedding2) ** 0.5
        return dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0
    
    def add_document(self, text: str, metadata: Dict[str, Any] = None) -> str:
        """Add a document to the knowledge base"""
        # Generate a document ID
        doc_id = f"doc_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Create embedding
        embedding = self._create_embedding(text)
        
        # Prepare data for storage
        data = {
            'id': doc_id,
            'text': text,
            'metadata': json.dumps(metadata or {}),
            'embedding': json.dumps(embedding),
            'created_at': datetime.now().isoformat(),
        }
        
        # Load existing data or create new DataFrame
        if os.path.exists(self.embedding_path):
            df = pd.read_excel(self.embedding_path)
            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
        else:
            df = pd.DataFrame([data])
        
        # Save to Excel
        df.to_excel(self.embedding_path, index=False)
        
        return doc_id
    
    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar documents in the knowledge base"""
        # Debug output
        print(f"\nEmbedding Search: Query = '{query}', top_k = {top_k}")
        print(f"Knowledge base path: {self.embedding_path}")
        print(f"Knowledge base exists: {self.embedding_path.exists()}")
        
        # Check if knowledge base file exists
        if not self.embedding_path.exists():
            print("Knowledge base file does not exist!")
            return []
        
        try:
            # Create query embedding
            query_embedding = self._create_embedding(query)
            
            # Load knowledge base
            df = pd.read_excel(self.embedding_path)
            print(f"Loaded knowledge base with {len(df)} documents")
            
            # If dataframe is empty, return empty list
            if df.empty:
                print("Knowledge base is empty!")
                return []
            
            # Calculate similarities
            similarities = []
            for _, row in df.iterrows():
                try:
                    # Load embedding - handle potential JSON parsing issues
                    if isinstance(row['embedding'], str):
                        doc_embedding = json.loads(row['embedding'])
                    else:
                        doc_embedding = row['embedding']
                        
                    # Load metadata
                    if isinstance(row['metadata'], str):
                        metadata = json.loads(row['metadata'])
                    else:
                        metadata = row['metadata'] or {}
                    
                    # Calculate similarity
                    similarity = self._cosine_similarity(query_embedding, doc_embedding)
                    
                    # Add to results
                    similarities.append({
                        'id': row['id'],
                        'text': row['text'],
                        'metadata': metadata,
                        'similarity': similarity
                    })
                except Exception as doc_error:
                    print(f"Error processing document {row.get('id', 'unknown')}: {doc_error}")
                    continue
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            results = similarities[:top_k]
            
            # Debug output for results
            print(f"Found {len(similarities)} documents with similarity scores")
            print(f"Returning top {len(results)} results")
            for i, res in enumerate(results):
                title = res['metadata'].get('title', 'No title')
                print(f"Result {i+1}: {title} (score: {res['similarity']:.4f})")
            
            return results
            
        except Exception as e:
            # Return empty list if there's any issue reading the knowledge base
            import traceback
            print(f"Error searching knowledge base: {e}")
            print(traceback.format_exc())
            return []
    
    def get_all_documents(self) -> pd.DataFrame:
        """Get all documents in the knowledge base"""
        if not self.embedding_path.exists():
            return pd.DataFrame(columns=['id', 'text', 'metadata', 'created_at'])
        
        try:
            df = pd.read_excel(self.embedding_path)
            # Remove embedding column to save space when displaying
            if 'embedding' in df.columns:
                df = df.drop(columns=['embedding'])
            return df
        except Exception as e:
            print(f"Error reading knowledge base: {e}")
            return pd.DataFrame(columns=['id', 'text', 'metadata', 'created_at'])
    
    def clear_knowledge_base(self) -> None:
        """Clear all documents from the knowledge base"""
        if os.path.exists(self.embedding_path):
            os.remove(self.embedding_path)

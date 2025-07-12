"""
1_Embeddings.py - Demonstration of embeddings and vector similarity

© 2025 NextRun Digital. All Rights Reserved.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import plotly.express as px
import json
import sys
from pathlib import Path
import time

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from utils.embedding import EmbeddingManager
from utils.config import configure_openai, get_openai_api_key
from openai import OpenAI

def main():
    # Initialize session state
    if "embeddings" not in st.session_state:
        st.session_state.embeddings = []
        
    st.title("Embeddings: Meaning as Math")
    st.markdown("© 2025 NextRun Digital. All Rights Reserved.")
    
    # Check if OpenAI API key is set
    if not configure_openai():
        st.warning("Please set your OpenAI API key in the home page to use this feature.")
        return
    
    # Initialize OpenAI client
    client = OpenAI(api_key=get_openai_api_key())
    
    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["What are Embeddings?", "Embedding Demo", "Visualize Embeddings"])
    
    with tab1:
        st.header("What are Embeddings?")
        
        st.markdown("""
        Embeddings are mathematical representations of meaning in a high-dimensional space.
        
        ### The Problem:
        - Machines don't "understand" words, images, or data.
        - We need to convert meaning into a format that machines can work with.
        
        ### The Solution:
        - Embeddings convert text, images, and data into vectors (lists of numbers).
        - Similar concepts are positioned close together in the embedding space.
        - This allows machines to understand relationships and similarities.
        
        ### Applications:
        - Semantic search
        - Recommendation systems
        - Document classification
        - RAG (Retrieval Augmented Generation)
        
        ### Example:
        - "dog" is closer to "puppy" than to "car" in embedding space
        - This means the machine can understand that dogs and puppies are related concepts
        """)
        
        st.subheader("How Embeddings Work")
        st.markdown("""
        1. Text is processed through a neural network trained on vast corpora of text.
        2. The network learns to encode semantic meaning into a fixed-length vector.
        3. For OpenAI's text-embedding-ada-002, each embedding has 1,536 dimensions.
        4. These high-dimensional vectors capture nuanced relationships between concepts.
        """)
        
        st.code("""
        # Python code to create an embedding with OpenAI
        from openai import OpenAI

        client = OpenAI(api_key="your-api-key")
        
        response = client.embeddings.create(
            input="Your text to be embedded",
            model="text-embedding-ada-002"
        )
        
        embedding = response.data[0].embedding
        # embedding is now a list of 1,536 floating point numbers
        """, language="python")
    
    with tab2:
        st.header("Embedding Demo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Create Embeddings")
            text_input = st.text_area("Enter text to create embedding", 
                                    "Embeddings convert text into mathematical representations.")
            
            if st.button("Generate Embedding"):
                with st.spinner("Generating embedding..."):
                    try:
                        response = client.embeddings.create(
                            input=text_input,
                            model="text-embedding-ada-002"
                        )
                        embedding = response.data[0].embedding
                        
                        # Store in session state
                        if "embeddings" not in st.session_state:
                            st.session_state.embeddings = []
                        
                        st.session_state.embeddings.append({
                            "text": text_input,
                            "embedding": embedding
                        })
                        
                        st.success(f"Embedding created! Vector length: {len(embedding)}")
                        
                        # Display first few dimensions
                        st.write("First 10 dimensions:")
                        st.write(embedding[:10])
                    except Exception as e:
                        st.error(f"Error generating embedding: {str(e)}")
        
        with col2:
            st.subheader("Compare Similarity")
            if "embeddings" in st.session_state and len(st.session_state.embeddings) >= 2:
                # Create dropdowns for selecting texts to compare
                texts = [item["text"] for item in st.session_state.embeddings]
                
                text1_idx = st.selectbox("Select first text", range(len(texts)), 
                                       format_func=lambda i: texts[i][:50] + "...")
                text2_idx = st.selectbox("Select second text", range(len(texts)), 
                                       format_func=lambda i: texts[i][:50] + "...")
                
                if st.button("Calculate Similarity"):
                    embedding1 = st.session_state.embeddings[text1_idx]["embedding"]
                    embedding2 = st.session_state.embeddings[text2_idx]["embedding"]
                    
                    # Calculate cosine similarity
                    dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
                    norm1 = sum(a * a for a in embedding1) ** 0.5
                    norm2 = sum(b * b for b in embedding2) ** 0.5
                    similarity = dot_product / (norm1 * norm2)
                    
                    st.metric("Cosine Similarity", f"{similarity:.4f}")
                    
                    # Interpret similarity
                    if similarity > 0.9:
                        st.success("These texts are very similar in meaning!")
                    elif similarity > 0.7:
                        st.info("These texts are somewhat similar.")
                    else:
                        st.warning("These texts are not very similar.")
            else:
                st.info("Generate at least 2 embeddings to compare similarity")
    
    with tab3:
        st.header("Visualize Embeddings")
        
        # Make sure embeddings exist and there's at least one
        if len(st.session_state.embeddings) > 0:
            st.subheader("Visualization")
            st.write("This visualization shows how the embeddings relate to each other in 2D space.")
            
            # Convert embeddings list to numpy array for compatibility with sklearn
            embedding_vectors = np.array([item["embedding"] for item in st.session_state.embeddings])
            texts = [item["text"][:30] + "..." if len(item["text"]) > 30 else item["text"] 
                    for item in st.session_state.embeddings]
            
            # Check if we have enough embeddings for t-SNE
            if len(embedding_vectors) < 3:
                st.warning("Need at least 3 embeddings for t-SNE visualization. Using PCA instead.")
                # Use PCA for very small datasets
                from sklearn.decomposition import PCA
                with st.spinner("Performing dimensionality reduction with PCA..."):
                    pca = PCA(n_components=2)
                    reduced_embeddings = pca.fit_transform(embedding_vectors)
            else:
                # Perform dimensionality reduction with t-SNE
                with st.spinner("Performing dimensionality reduction with t-SNE..."):
                    # Set perplexity to about 1/3 of the dataset size, but not less than 2
                    perplexity = max(2, min(30, len(embedding_vectors) // 3))
                    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
                    reduced_embeddings = tsne.fit_transform(embedding_vectors)
            
            # Create DataFrame for plotting
            df = pd.DataFrame({
                'x': reduced_embeddings[:, 0],
                'y': reduced_embeddings[:, 1],
                'text': texts
            })
            
            # Plot with Plotly
            fig = px.scatter(
                df, x='x', y='y', text='text',
                title='2D Visualization of Text Embeddings'
            )
            fig.update_traces(textposition='top center')
            st.plotly_chart(fig, use_container_width=True)
            
            st.caption("Note: This visualization reduces the 1,536-dimensional embeddings to 2D, so distances are approximate.")
        else:
            st.info("Generate at least 3 embeddings to visualize")
        
        # Option to clear embeddings
        if "embeddings" in st.session_state and len(st.session_state.embeddings) > 0:
            if st.button("Clear All Embeddings"):
                st.session_state.embeddings = []
                st.success("All embeddings cleared!")

if __name__ == "__main__":
    main()

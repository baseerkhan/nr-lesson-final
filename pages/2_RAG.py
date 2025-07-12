"""
2_RAG.py - Demonstration of Retrieval Augmented Generation

© 2025 NextRun Digital. All Rights Reserved.
"""

import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
import time
import json

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from utils.rag import RAGSystem
from utils.embedding import EmbeddingManager
from utils.config import configure_openai, DATA_DIR

def main():
    st.title("Retrieval Augmented Generation (RAG)")
    st.markdown("© 2025 NextRun Digital. All Rights Reserved.")
    
    # Check if OpenAI API key is set
    if not configure_openai():
        st.warning("Please set your OpenAI API key in the home page to use this feature.")
        return
    
    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["What is RAG?", "RAG Demo", "Build Your Knowledge Base"])
    
    with tab1:
        st.header("What is Retrieval Augmented Generation?")
        
        st.markdown("""
        Retrieval Augmented Generation (RAG) is a technique that enhances LLM responses by retrieving
        relevant external knowledge before generating an answer.
        
        ### The RAG Process:
        
        1. **Retrieval**: The system searches an external knowledge source (like a vector database)
           to find relevant information based on the user query.
        
        2. **Augmentation**: The retrieved information is combined with the original user query
           to create a more informative prompt.
        
        3. **Generation**: The LLM processes the augmented prompt and generates a response that
           integrates both its pre-trained knowledge and the retrieved information.
        
        ### Benefits of RAG:
        
        - **Improved Accuracy**: Access to specific, up-to-date information
        - **Reduced Hallucinations**: Grounds responses in factual context
        - **Access to Private Data**: Can reference your organization's proprietary information
        - **Real-time Knowledge**: Can incorporate recent information not in the LLM's training data
        """)
        
        st.subheader("How RAG Works")
        st.image("https://miro.medium.com/v2/resize:fit:1400/1*HU_FjnDJqs2BRLIzQQZsDA.png", 
                caption="RAG Architecture (Source: Medium)")
        
        st.markdown("""
        ### Implementation Components:
        
        1. **Document Processing**: Breaking documents into chunks and creating embeddings
        2. **Vector Storage**: Storing document chunks and their embeddings
        3. **Retrieval System**: Finding the most relevant chunks for a query
        4. **Context Engineering**: Formatting retrieved content for the LLM
        5. **Response Generation**: Having the LLM generate answers based on the context
        """)
        
        st.code("""
        # Python code to implement a simple RAG system
        from openai import OpenAI
        
        # 1. Retrieve relevant documents
        similar_docs = knowledge_base.search(user_query, top_k=3)
        
        # 2. Prepare context for the LLM
        context = "\\n\\n".join([doc.text for doc in similar_docs])
        
        # 3. Create augmented prompt
        system_prompt = f\"\"\"
        Answer based on the following context:
        {context}
        \"\"\"
        
        # 4. Generate response with LLM
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ]
        )
        """, language="python")
    
    with tab2:
        st.header("RAG Demo")
        
        # Initialize RAG system with debugging
        embedding_manager = EmbeddingManager()
        
        # Debug: Check knowledge base status
        kb_path = DATA_DIR / "knowledge_base.xlsx"
        st.write(f"Debug - Knowledge base path: {kb_path}")
        st.write(f"Debug - Knowledge base exists: {os.path.exists(kb_path)}")
        
        # Show documents in knowledge base
        docs = embedding_manager.get_all_documents()
        st.write(f"Debug - Documents in knowledge base: {len(docs)}")
        
        if not docs.empty:
            st.write("Debug - First few documents in knowledge base:")
            st.dataframe(docs.head(3))
        else:
            st.error("Debug - No documents found in knowledge base!")
        
        rag_system = RAGSystem(embedding_manager)
        
        # Create columns for query and results
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Ask a Question")
            user_query = st.text_area("Enter your question", 
                                    "What are the key concepts of augmented LLMs?")
            
            top_k = st.slider("Number of documents to retrieve", min_value=1, max_value=5, value=3)
            
            if st.button("Submit Query"):
                if user_query:
                    # Check if there are documents in the knowledge base
                    doc_count = len(embedding_manager.get_all_documents())
                    st.write(f"Debug - Document count before query: {doc_count}")
                    
                    if doc_count == 0:
                        st.error("No documents in the knowledge base. Please add documents in the 'Manage Knowledge Base' tab before querying.")
                    else:
                        # Show debug information in an expander
                        with st.expander("Debug Information", expanded=True):
                            st.write(f"Processing query: '{user_query}'")
                            st.write(f"Looking for top {top_k} relevant documents")
                        
                        with st.spinner("Processing your question..."):
                            # Debug: Try direct document retrieval
                            debug_docs = embedding_manager.search(user_query, top_k=top_k)
                            
                            with st.expander("Direct Search Debug", expanded=True):
                                st.write(f"Direct search found {len(debug_docs)} documents")
                                for i, doc in enumerate(debug_docs):
                                    st.write(f"Document {i+1} - Score: {doc['similarity']:.4f}")
                                    st.write(f"Text: {doc['text'][:100]}...")
                                    if 'metadata' in doc and doc['metadata']:
                                        st.write(f"Title: {doc['metadata'].get('title', 'No title')}")
                            
                            # Process the query with RAG
                            result = rag_system.query(user_query, top_k=top_k)
                            
                            # Store result in session state
                            st.session_state.rag_result = result
                            
                            st.success("Query processed successfully!")
                else:
                    st.error("Please enter a question.")
        
        with col2:
            st.subheader("Response")
            if "rag_result" in st.session_state:
                result = st.session_state.rag_result
                
                st.markdown("### Answer:")
                st.write(result["answer"])
                
                # Debug information for context
                with st.expander("Debug - Context Analysis", expanded=True):
                    st.markdown("### Context Debug")
                    
                    # Check if we have any sources
                    if result["sources"]:
                        st.success(f"✅ Found {len(result['sources'])} relevant documents")
                        
                        # Check if any documents are about augmented LLMs
                        llm_related = False
                        for src in result["sources"]:
                            title = src.get('metadata', {}).get('title', '').lower()
                            text = src.get('text', '').lower()
                            
                            if 'augmented llm' in title or 'augmented llm' in text:
                                llm_related = True
                                st.success(f"✅ Found document related to augmented LLMs: {title}")
                                
                        if not llm_related:
                            st.warning("⚠️ No documents specifically about augmented LLMs found")
                            
                        # Show top document details
                        top_doc = result["sources"][0]
                        top_title = top_doc.get('metadata', {}).get('title', 'No title')
                        top_score = top_doc.get('similarity', 0)
                        
                        st.write(f"Top document: '{top_title}' with similarity score {top_score:.4f}")
                        st.write(f"Full text: {top_doc.get('text')}")
                    else:
                        st.error("❌ No relevant documents found!")
                
                with st.expander("View Retrieved Context"):
                    st.markdown("### Retrieved Documents:")
                    st.write(result["context"])
                
                with st.expander("View Sources"):
                    if result["sources"]:
                        for i, source in enumerate(result["sources"]):
                            st.markdown(f"**Document {i+1}**")
                            st.write(f"Relevance Score: {source['similarity']:.4f}")
                            if source.get('metadata'):
                                st.write(f"Metadata: {source['metadata']}")
                            # Show full text for debugging
                            with st.expander(f"Full text of Document {i+1}"):
                                st.write(source.get('text', 'No text available'))
                    else:
                        st.warning("No relevant documents found in knowledge base")
                
                # Add LLM conversation details
                with st.expander("View LLM Conversation"):
                    st.markdown("### System Prompt")
                    st.code(result["conversation"]["system_prompt"], language="")
                    
                    st.markdown("### User Query")
                    st.code(result["conversation"]["messages"][1]["content"], language="")
                    
                    st.markdown("### LLM Response")
                    st.code(result["conversation"]["raw_response"]["choices"][0]["message"]["content"], language="")
                    
                    st.markdown("### Usage Statistics")
                    usage = result["conversation"]["raw_response"].get("usage", {})
                    st.json(usage)
                    
                    st.markdown("### Complete Raw Response")
                    st.json(result["conversation"]["raw_response"])
            else:
                st.info("Submit a query to see results")
    
    with tab3:
        st.header("Build Your Knowledge Base")
        
        st.markdown("""
        Add documents to your knowledge base to improve RAG responses.
        These documents will be processed into embeddings and stored for retrieval.
        """)
        
        # Initialize embedding manager
        embedding_manager = EmbeddingManager()
        
        # Text input for new documents
        st.subheader("Add New Document")
        document_text = st.text_area("Document content", height=150)
        document_title = st.text_input("Document title")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            if st.button("Add to Knowledge Base"):
                if document_text and document_title:
                    with st.spinner("Processing document..."):
                        # Add metadata
                        metadata = {"title": document_title, "added_date": time.strftime("%Y-%m-%d %H:%M:%S")}
                        
                        # Add to knowledge base
                        doc_id = embedding_manager.add_document(document_text, metadata)
                        
                        st.success(f"Document added to knowledge base with ID: {doc_id}")
                else:
                    st.error("Please enter both document content and title.")
        
        with col2:
            if st.button("Clear Knowledge Base"):
                embedding_manager.clear_knowledge_base()
                st.success("Knowledge base cleared!")
        
        # View existing documents
        st.subheader("Knowledge Base Contents")
        docs_df = embedding_manager.get_all_documents()
        
        if not docs_df.empty:
            st.dataframe(docs_df[['id', 'text', 'metadata', 'created_at']], use_container_width=True)
        else:
            st.info("No documents in knowledge base yet.")
            
            # Add sample documents
            if st.button("Add Sample Documents"):
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
                    }
                ]
                
                with st.spinner("Adding sample documents..."):
                    for doc in sample_docs:
                        metadata = {"title": doc["title"], "added_date": time.strftime("%Y-%m-%d %H:%M:%S")}
                        embedding_manager.add_document(doc["text"], metadata)
                
                st.success(f"Added {len(sample_docs)} sample documents to knowledge base!")
                st.rerun()

if __name__ == "__main__":
    main()

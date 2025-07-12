"""
skip_api.py - Utilities for running the app without OpenAI API access

© 2025 NextRun Digital. All Rights Reserved.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from .config import DATA_DIR

def initialize_empty_knowledge_base():
    """Create empty knowledge base file to avoid errors"""
    kb_path = DATA_DIR / "knowledge_base.xlsx"
    
    if not kb_path.exists():
        # Create empty dataframe with required columns
        df = pd.DataFrame(columns=[
            "text", "embedding", "metadata"
        ])
        # Save to Excel
        df.to_excel(kb_path, index=False)
        print(f"Created empty knowledge base file at {kb_path}")
        
    return True

def initialize_empty_memory_file():
    """Create empty memory file to avoid errors"""
    memory_path = DATA_DIR / "memory.xlsx"
    
    if not memory_path.exists():
        # Create empty dataframe with required columns
        df = pd.DataFrame(columns=[
            "timestamp", "role", "content", "embedding"
        ])
        # Save to Excel
        df.to_excel(memory_path, index=False)
        print(f"Created empty memory file at {memory_path}")
        
    return True

def manual_initialization_ui():
    """UI for manual initialization of knowledge base"""
    st.warning("⚠️ Running in limited mode without OpenAI API access")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Initialize Empty Knowledge Base"):
            initialize_empty_knowledge_base()
            st.success("✅ Created empty knowledge base file")
            st.info("You'll need to add documents manually when an OpenAI API key is configured.")
            
    with col2:
        if st.button("Initialize Empty Memory File"):
            initialize_empty_memory_file()
            st.success("✅ Created empty memory file")
            st.info("Memory will be stored once an OpenAI API key is configured.")

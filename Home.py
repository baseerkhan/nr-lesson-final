"""
Home.py - Main entry point for AI Leadership Course Application

© 2025 NextRun Digital. All Rights Reserved.
"""

import streamlit as st
import os
import sys
from pathlib import Path

# Add the project directory to Python path
sys.path.append(str(Path(__file__).parent))

# Import utilities
from utils import configure_openai, check_mcp_server, get_openai_api_key
from utils.skip_api import manual_initialization_ui

# Set page configuration
st.set_page_config(
    page_title="AI Leadership Course",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    # Initialize data directory
    from utils.data_loader import ensure_data_directory
    ensure_data_directory()
    
    # Check if OpenAI API key is configured
    api_configured = configure_openai()
    data_initialized = False
    
    if api_configured:
        try:
            from utils.data_loader import initialize_data
            data_initialized = initialize_data()
            print("DEBUG: Data initialization attempted, result:", data_initialized)
        except Exception as e:
            print(f"DEBUG: Error during data initialization: {str(e)}")
            st.error(f"Error during data initialization: {str(e)}")
    
    # Header
    st.title("Engineer to AI Leader in One Weekend")
    st.subheader("Modern AI Systems: Concepts and Implementation")
    
    # Copyright notice
    st.markdown("© 2025 NextRun Digital. All Rights Reserved.")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Introduction
        st.markdown("""
        ## Welcome to the AI Leadership Course
        
        This interactive application demonstrates the key concepts and technologies powering modern AI systems.
        
        ### What You'll Learn:
        
        1. **Augmented LLM** - How LLMs can be enhanced with additional capabilities
        2. **Memory & Context Engineering** - Building persistent memory for AI systems
        3. **Retrieval Augmented Generation (RAG)** - Using external knowledge to improve LLM responses
        4. **Embeddings** - Converting meaning into mathematical representations
        5. **Tool Calling** - Enabling LLMs to use external tools and APIs
        6. **Model Context Protocol (MCP)** - Standardizing tool integration
        
        ### How to Use This Application:
        
        Navigate through the different sections using the sidebar menu. Each page contains:
        
        - **Explanation** of the concept
        - **Interactive demo** to see it in action
        - **Code examples** showing how to implement it
        """)
    
    with col2:
        st.image("https://miro.medium.com/v2/resize:fit:1400/1*3PpStGtYQFStRDFJCYpHFQ.png", 
                caption="Augmented LLM Architecture")
        
        if data_initialized:
            st.success("✅ Sample data initialized successfully!")
        elif not api_configured:
            # Show manual initialization options if no API key
            manual_initialization_ui()
        
        # Quick navigation
        st.subheader("Quick Navigation")
        st.markdown("""
        - [Embeddings](/Embeddings)
        - [RAG](/RAG)
        - [Memory](/Memory)
        - [Tool Calling](/ToolCalling)
        - [MCP](/MCP)
        """)
    
    # System status
    st.divider()
    st.subheader("System Status")
    
    col1, col2 = st.columns(2)
    with col1:
        # Check API key
        api_key = get_openai_api_key()
        if api_key:
            masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "****"
            st.success(f"✅ OpenAI API Key configured: {masked_key}")
        else:
            st.error(" OpenAI API Key not configured")
            st.info("Set API key in sidebar to enable all features")
    
    with col2:
        # Check MCP server
        mcp_status = check_mcp_server()
        if mcp_status:
            st.success("✅ MCP Server is running")
        else:
            st.error("❌ MCP Server is not running")
            st.info("Run 'python start_mcp_server.py' to start the integrated MCP server")
    
    # Getting started
    st.divider()
    st.subheader("Getting Started")
    st.markdown("""
    1. Set your OpenAI API key in the sidebar
    2. Start the MCP server by running `python start_mcp_server.py`
    3. Navigate to the different pages using the sidebar menu
    """)
    
    # Sidebar for API key
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("OpenAI API Key", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.session_state.openai_api_key = api_key
            st.success("API Key set successfully!")
        else:
            st.warning("Please enter your OpenAI API Key to use all features")
            
        # API mode toggle
        st.divider()
        st.header("API Mode Settings")
        use_api = st.toggle("Use OpenAI API", value=True, help="Turn off to use fallback mode without API calls")
        st.session_state["use_api_mode"] = use_api
        
        if not use_api:
            st.info("⚠️ Running in fallback mode - no API calls will be made")
            st.warning("Responses will be limited in quality but won't consume API quota")
        else:
            st.success("✅ Using OpenAI API for all operations")
            if api_key:
                st.info("Make sure your API key has sufficient quota")
            else:
                st.warning("API key required when API mode is enabled")
        
        # Check for MCP server
        st.divider()
        st.header("MCP Server Status")
        if check_mcp_server():
            st.success("✅ MCP Server is running")
        else:
            st.error("❌ MCP Server is not running")
            st.info("Run 'python start_mcp_server.py' to start the integrated MCP server")

if __name__ == "__main__":
    main()

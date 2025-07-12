"""
config.py - Configuration utilities for the AI Leadership Course application

© 2025 NextRun Digital. All Rights Reserved.
"""

import os
from pathlib import Path
import openai
import streamlit as st

# Base directories
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"

# MCP Server settings
MCP_SERVER_URL = "http://127.0.0.1:8000"

def get_openai_api_key():
    """Get OpenAI API key from environment or session state"""
    # First check session state
    if hasattr(st.session_state, "openai_api_key") and st.session_state.openai_api_key:
        key = st.session_state.openai_api_key
        print(f"DEBUG: Using API key from session state: {key[:4]}...{key[-4:] if len(key) > 8 else ''}")
        return key
    
    # Then check environment variable
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        print(f"DEBUG: Using API key from environment variable: {api_key[:4]}...{api_key[-4:] if len(api_key) > 8 else ''}")
        return api_key
        
    print("DEBUG: No OpenAI API key found!")
    return None

def configure_openai():
    """Configure OpenAI with API key"""
    api_key = get_openai_api_key()
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        print(f"DEBUG: Successfully configured OpenAI with key: {api_key[:4]}...{api_key[-4:] if len(api_key) > 8 else ''}")
        return True
    print("DEBUG: Failed to configure OpenAI - no valid API key found")
    return False

def check_mcp_server():
    """Check if MCP server is running"""
    try:
        import requests
        response = requests.get(f"{MCP_SERVER_URL}/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False

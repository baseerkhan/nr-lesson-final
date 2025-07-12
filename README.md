# AI Leadership Course Application

© 2025 NextRun Digital. All Rights Reserved.

This application demonstrates key concepts in modern AI systems including:

1. Augmented LLM - How LLMs can be enhanced with additional capabilities
2. Memory & Context Engineering - Building persistent memory for AI systems
3. Retrieval Augmented Generation (RAG) - Using external knowledge to improve LLM responses
4. Embeddings - Converting meaning into mathematical representations
5. Tool Calling - Enabling LLMs to use external tools and APIs
6. Model Context Protocol (MCP) - Standardizing tool integration

## Setup

1. Clone this repository
2. Install requirements:
```bash
pip install -r requirements.txt
```
3. Set up your OpenAI API key as an environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```
4. Start the integrated MCP server:
```bash
python start_mcp_server.py
```
5. In a separate terminal, run the Streamlit application:
```bash
streamlit run Home.py
```

## Structure

- `Home.py` - Main entry point for the application
- `pages/` - Individual demonstration pages for each concept
- `utils/` - Helper functions and utilities
- `data/` - Sample data for demonstrations
- `mcpserver/` - Integrated MCP server for tool calling demonstrations

## Included MCP Tools

The integrated MCP server provides several tools that can be called by the LLM:

1. **get_current_time** - Get current time in a specified timezone
2. **calculate_age** - Calculate age based on a birth date
3. **get_weather** - Get simulated weather data for a location
4. **search_products** - Search a simulated product database
5. **do_math_calculation** - Safely evaluate mathematical expressions

## Requirements

- Python 3.8+
- OpenAI API key
- FastAPI and Uvicorn (installed via requirements.txt)

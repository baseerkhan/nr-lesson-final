"""
main.py - MCP server implementation

© 2025 NextRun Digital. All Rights Reserved.
"""

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import logging
from pathlib import Path

from .services import get_current_time, calculate_age, get_weather, search_products, do_math_calculation
from .schema import Tool, ToolCallRequest, ToolCallResponse

# Create FastAPI app
app = FastAPI(
    title="MCP Server",
    description="Model Context Protocol Server for AI Leadership Course",
    version="1.0.0",
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mcpserver")

# Store registered tools
tools_registry: Dict[str, Dict] = {}

# Register tools
def register_tool(name: str, description: str, parameters: Dict[str, Any], function):
    """Register a tool with the MCP server"""
    tools_registry[name] = {
        "name": name,
        "description": description,
        "parameters": parameters,
        "function": function
    }
    logger.info(f"Registered tool: {name}")

# Register available tools
register_tool(
    name="get_current_time",
    description="Get the current time in a specified timezone",
    parameters={
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": "Timezone name (e.g., 'America/New_York')"
            }
        }
    },
    function=get_current_time
)

register_tool(
    name="calculate_age",
    description="Calculate age based on birth date",
    parameters={
        "type": "object",
        "properties": {
            "birth_date": {
                "type": "string",
                "description": "Birth date in YYYY-MM-DD format"
            }
        },
        "required": ["birth_date"]
    },
    function=calculate_age
)

register_tool(
    name="get_weather",
    description="Get current weather for a location",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name, e.g. 'San Francisco'"
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "default": "celsius"
            }
        },
        "required": ["location"]
    },
    function=get_weather
)

register_tool(
    name="search_products",
    description="Search for products in a catalog",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            },
            "category": {
                "type": "string",
                "description": "Product category"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "default": 5
            }
        },
        "required": ["query"]
    },
    function=search_products
)

register_tool(
    name="do_math_calculation",
    description="Perform a mathematical calculation",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Mathematical expression to evaluate (e.g., '2 + 2')"
            }
        },
        "required": ["expression"]
    },
    function=do_math_calculation
)

@app.get("/")
def read_root():
    return {"message": "MCP Server is running"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/tools", response_model=List[Tool])
def list_tools():
    """List all available tools"""
    result = []
    for name, tool in tools_registry.items():
        result.append({
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["parameters"]
        })
    return result

@app.get("/tools/{tool_name}", response_model=Tool)
def get_tool(tool_name: str):
    """Get details about a specific tool"""
    if tool_name not in tools_registry:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
    
    tool = tools_registry[tool_name]
    return {
        "name": tool["name"],
        "description": tool["description"],
        "parameters": tool["parameters"]
    }

@app.post("/tools/{tool_name}/call", response_model=ToolCallResponse)
def call_tool(tool_name: str, request: ToolCallRequest):
    """Call a specific tool with parameters"""
    if tool_name not in tools_registry:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
    
    tool = tools_registry[tool_name]
    function = tool["function"]
    
    try:
        # Call the function with parameters
        result = function(**request.parameters)
        return {"result": result}
    except Exception as e:
        logger.error(f"Error calling tool {tool_name}: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error calling tool: {str(e)}")

def start_server(host="127.0.0.1", port=8000):
    """Start the MCP server"""
    uvicorn.run("mcpserver.main:app", host=host, port=port, reload=True)

if __name__ == "__main__":
    start_server()

"""
tool_calling.py - Utilities for tool calling and MCP integration

© 2025 NextRun Digital. All Rights Reserved.
"""

import requests
import json
from typing import Dict, Any, List, Optional
import streamlit as st
from .config import MCP_SERVER_URL, get_openai_api_key
from openai import OpenAI

class MCPToolCaller:
    """Interface for calling tools via the MCP server"""
    
    def __init__(self, server_url: str = MCP_SERVER_URL):
        """Initialize the MCP tool caller"""
        self.server_url = server_url
        api_key = get_openai_api_key()
        # Initialize the OpenAI client with API key
        self.client = OpenAI(api_key=api_key)
        
    def list_available_tools(self) -> List[Dict[str, Any]]:
        """Get a list of available tools from the MCP server"""
        try:
            response = requests.get(f"{self.server_url}/tools")
            if response.status_code == 200:
                return response.json()
            return []
        except Exception as e:
            st.error(f"Error connecting to MCP server: {e}")
            return []
    
    def call_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Call a tool on the MCP server with given parameters"""
        try:
            payload = {
                "parameters": parameters
            }
            response = requests.post(
                f"{self.server_url}/tools/{tool_name}/call",
                json=payload
            )
            if response.status_code == 200:
                return response.json().get("result", {})
            else:
                return {"error": f"Error calling tool: {response.text}", "success": False}
        except Exception as e:
            return {"error": str(e), "success": False}
    
    def process_with_tool_calling(self, user_query: str, system_prompt: str = None) -> Dict[str, Any]:
        """Process a query with OpenAI's tool calling feature and MCP integration"""
        # First, get available tools from MCP server
        tools = self.list_available_tools()
        
        if not tools:
            return {
                "response": "No tools available from MCP server",
                "tool_calls": [],
                "tool_results": []
            }
        
        # Format tools for OpenAI API
        openai_tools = []
        for tool in tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {})
                }
            })
        
        # Default system prompt if none provided
        if not system_prompt:
            system_prompt = """
            You are an AI assistant with the ability to use external tools.
            Based on the user's request, you can call tools to help answer their question.
            If a tool call is needed, please use the provided tools.
            """
        
        # Make the initial request to OpenAI
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query}
            ],
            tools=openai_tools,
            tool_choice="auto",
        )
        
        response_message = response.choices[0].message
        tool_calls = response_message.tool_calls
        
        # If no tool calls were made, return the response
        if not tool_calls:
            return {
                "response": response_message.content,
                "tool_calls": [],
                "tool_results": []
            }
        
        # Process tool calls
        tool_results = []
        follow_up_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
            {"role": "assistant", "content": response_message.content, "tool_calls": tool_calls}
        ]
        
        for tool_call in tool_calls:
            function_name = tool_call.function.name
            function_args = json.loads(tool_call.function.arguments)
            
            # Call the MCP tool
            result = self.call_tool(function_name, function_args)
            
            tool_results.append({
                "name": function_name,
                "arguments": function_args,
                "result": result
            })
            
            # Add tool result to messages
            follow_up_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "name": function_name,
                "content": json.dumps(result)
            })
        
        # Get a final response that takes into account the tool results
        final_response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=follow_up_messages,
        )
        
        return {
            "response": final_response.choices[0].message.content,
            "tool_calls": tool_calls,
            "tool_results": tool_results
        }

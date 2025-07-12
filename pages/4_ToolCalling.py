"""
4_ToolCalling.py - Demonstration of Tool and Function Calling with LLMs

© 2025 NextRun Digital. All Rights Reserved.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import time
import json
import requests
from typing import Dict, Any

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from utils.tool_calling import MCPToolCaller
from utils.config import configure_openai, MCP_SERVER_URL, check_mcp_server, get_openai_api_key
from openai import OpenAI

def main():
    st.title("Tool and Function Calling")
    st.markdown("© 2025 NextRun Digital. All Rights Reserved.")
    
    # Check if OpenAI API key is set
    if not configure_openai():
        st.warning("Please set your OpenAI API key in the home page to use this feature.")
        return
    
    # Check MCP server status
    mcp_running = check_mcp_server()
    
    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Tool Calling Concepts", "Function Calling Demo", "MCP Integration"])
    
    with tab1:
        st.header("Tool and Function Calling")
        
        st.markdown("""
        ### What is Tool or Function Calling?
        
        Tool or function calling lets an LLM call external tools or APIs to:
        - Fetch live data
        - Do calculations
        - Access databases
        - Trigger business processes
        
        ### How It Works:
        
        1. LLM generates a special output describing:
           - Which tool to call
           - What parameters to send
        
        2. The app or framework reads this output
        
        3. The tool is executed with the specified parameters
        
        4. Results are fed back to the LLM for further processing
        
        ### Applications:
        
        - Data analysis assistants
        - Booking systems
        - Productivity tools
        - Code generation
        - Automation agents
        """)
        
        st.subheader("Implementation")
        st.markdown("""
        OpenAI and other LLM providers have built-in support for function calling.
        The general approach is:
        
        1. Define the tools/functions available to the model
        2. Send these definitions along with the user query
        3. The LLM decides if and which function to call
        4. Execute the function and return the results to the LLM
        5. LLM incorporates the results into its final response
        """)
        
        st.code("""
        # Define functions/tools available to the model
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather in a location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA"
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["celsius", "fahrenheit"]
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
        
        # Ask the model, providing the tools
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": "What's the weather like in Boston?"}
            ],
            tools=tools,
            tool_choice="auto"
        )
        
        # Check if the model wants to call a function
        message = response.choices[0].message
        tool_calls = message.tool_calls
        
        if tool_calls:
            # Call the function
            function_name = tool_calls[0].function.name
            function_args = json.loads(tool_calls[0].function.arguments)
            
            if function_name == "get_weather":
                # Implement function logic
                weather_data = get_weather_data(function_args["location"])
                
                # Send function result back to model
                second_response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": "What's the weather like in Boston?"},
                        message,
                        {
                            "role": "tool",
                            "tool_call_id": tool_calls[0].id,
                            "name": "get_weather",
                            "content": json.dumps(weather_data)
                        }
                    ]
                )
                
                # Get the final response
                final_response = second_response.choices[0].message.content
        """, language="python")
    
    with tab2:
        st.header("Function Calling Demo")
        
        # Initialize OpenAI client
        client = OpenAI(api_key=get_openai_api_key())
        
        st.markdown("""
        This demo shows how an LLM can use function calling to perform tasks.
        We've defined three sample functions that the LLM can call:
        
        1. `get_weather` - Get weather information for a location
        2. `calculate_math` - Perform mathematical calculations
        3. `search_products` - Search for products in a catalog
        
        Try asking questions that might require these functions!
        """)
        
        # User input
        user_input = st.text_input("Ask something that might need a function call",
                                 "What's the weather in New York City?")
        
        if st.button("Process with Function Calling"):
            if user_input:
                with st.spinner("Processing..."):
                    try:
                        # Define functions
                        tools = [
                            {
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "description": "Get the current weather in a location",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "location": {
                                                "type": "string",
                                                "description": "The city and state, e.g. San Francisco, CA"
                                            },
                                            "unit": {
                                                "type": "string",
                                                "enum": ["celsius", "fahrenheit"],
                                                "description": "The temperature unit to use"
                                            }
                                        },
                                        "required": ["location"]
                                    }
                                }
                            },
                            {
                                "type": "function",
                                "function": {
                                    "name": "calculate_math",
                                    "description": "Calculate the result of a math expression",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "expression": {
                                                "type": "string",
                                                "description": "The mathematical expression to evaluate"
                                            }
                                        },
                                        "required": ["expression"]
                                    }
                                }
                            },
                            {
                                "type": "function",
                                "function": {
                                    "name": "search_products",
                                    "description": "Search for products in a catalog",
                                    "parameters": {
                                        "type": "object",
                                        "properties": {
                                            "query": {
                                                "type": "string",
                                                "description": "The search query"
                                            },
                                            "category": {
                                                "type": "string",
                                                "description": "Product category filter"
                                            },
                                            "max_results": {
                                                "type": "integer",
                                                "description": "Maximum number of results to return"
                                            }
                                        },
                                        "required": ["query"]
                                    }
                                }
                            }
                        ]
                        
                        # First API call to decide which function to call
                        response = client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "system", "content": "You are an AI assistant that can call functions to help answer questions."},
                                {"role": "user", "content": user_input}
                            ],
                            tools=tools,
                            tool_choice="auto"
                        )
                        
                        message = response.choices[0].message
                        st.write("### LLM Response:")
                        
                        if message.tool_calls:
                            # Display function call details
                            for tool_call in message.tool_calls:
                                function_name = tool_call.function.name
                                function_args = json.loads(tool_call.function.arguments)
                                
                                st.info(f"The LLM wants to call the function: **{function_name}**")
                                st.json(function_args)
                                
                                # Simulate function execution and results
                                function_result = {}
                                
                                if function_name == "get_weather":
                                    location = function_args.get("location", "")
                                    unit = function_args.get("unit", "celsius")
                                    
                                    function_result = {
                                        "location": location,
                                        "temperature": 72 if unit == "fahrenheit" else 22,
                                        "unit": unit,
                                        "condition": "sunny",
                                        "humidity": 45,
                                        "wind_speed": 10
                                    }
                                
                                elif function_name == "calculate_math":
                                    expression = function_args.get("expression", "")
                                    try:
                                        # SECURITY NOTE: In a production environment, you would need
                                        # to sanitize this input before evaluation
                                        import ast
                                        import operator
                                        
                                        # Define safe operations
                                        safe_operators = {
                                            ast.Add: operator.add,
                                            ast.Sub: operator.sub,
                                            ast.Mult: operator.mul,
                                            ast.Div: operator.truediv,
                                            ast.USub: operator.neg
                                        }
                                        
                                        def safe_eval(expr):
                                            """Safely evaluate a math expression"""
                                            parsed = ast.parse(expr, mode='eval').body
                                            
                                            def eval_node(node):
                                                if isinstance(node, ast.Num):
                                                    return node.n
                                                elif isinstance(node, ast.BinOp):
                                                    op_type = type(node.op)
                                                    if op_type not in safe_operators:
                                                        raise ValueError(f"Unsupported operation: {op_type}")
                                                    left = eval_node(node.left)
                                                    right = eval_node(node.right)
                                                    return safe_operators[op_type](left, right)
                                                elif isinstance(node, ast.UnaryOp):
                                                    op_type = type(node.op)
                                                    if op_type not in safe_operators:
                                                        raise ValueError(f"Unsupported operation: {op_type}")
                                                    operand = eval_node(node.operand)
                                                    return safe_operators[op_type](operand)
                                                else:
                                                    raise ValueError(f"Unsupported node type: {type(node)}")
                                            
                                            return eval_node(parsed)
                                        
                                        result = safe_eval(expression)
                                        function_result = {
                                            "expression": expression,
                                            "result": result
                                        }
                                    except Exception as e:
                                        function_result = {
                                            "expression": expression,
                                            "error": f"Could not evaluate: {str(e)}"
                                        }
                                
                                elif function_name == "search_products":
                                    query = function_args.get("query", "")
                                    category = function_args.get("category", "")
                                    max_results = function_args.get("max_results", 3)
                                    
                                    # Fake product database
                                    products = [
                                        {"id": 1, "name": "Laptop Pro", "category": "Electronics", "price": 1299.99},
                                        {"id": 2, "name": "Smartphone Ultra", "category": "Electronics", "price": 899.99},
                                        {"id": 3, "name": "Coffee Maker", "category": "Home", "price": 89.99},
                                        {"id": 4, "name": "Running Shoes", "category": "Sports", "price": 129.99},
                                        {"id": 5, "name": "Desk Lamp", "category": "Home", "price": 39.99}
                                    ]
                                    
                                    # Filter by category if provided
                                    if category:
                                        products = [p for p in products if p["category"].lower() == category.lower()]
                                    
                                    # Filter by query
                                    results = [p for p in products if query.lower() in p["name"].lower()]
                                    
                                    function_result = {
                                        "query": query,
                                        "category": category,
                                        "results": results[:max_results],
                                        "total_results": len(results)
                                    }
                                
                                st.write("### Function Result:")
                                st.json(function_result)
                                
                                # Get final response from LLM with function result
                                second_response = client.chat.completions.create(
                                    model="gpt-3.5-turbo",
                                    messages=[
                                        {"role": "system", "content": "You are an AI assistant that can call functions to help answer questions."},
                                        {"role": "user", "content": user_input},
                                        message,
                                        {
                                            "role": "tool",
                                            "tool_call_id": tool_call.id,
                                            "name": function_name,
                                            "content": json.dumps(function_result)
                                        }
                                    ]
                                )
                                
                                st.write("### Final Response:")
                                st.write(second_response.choices[0].message.content)
                        else:
                            # No function call needed
                            st.write(message.content)
                            st.info("No function calls were needed to answer this query.")
                    
                    except Exception as e:
                        st.error(f"Error during processing: {str(e)}")
            else:
                st.warning("Please enter a question.")
    
    with tab3:
        st.header("MCP Integration")
        
        if not mcp_running:
            st.error("MCP server is not running. Please start the MCP server to use this feature.")
            st.info("Run the following command to start the MCP server: `python -m nextrunmcp.main`")
        else:
            st.success("MCP server is running!")
            
            # Initialize MCP tool caller
            mcp_caller = MCPToolCaller()
            
            st.markdown("""
            ### Model Context Protocol (MCP)
            
            MCP provides a standardized way for LLMs to discover and call tools:
            
            - Defines tool names, input parameters, and output formats
            - Makes LLMs interoperable across different applications
            - Helps manage security and validation
            - Enables complex workflows through agent orchestration
            
            This demo shows integration with the NextRun MCP server.
            """)
            
            # Display available tools
            tools = mcp_caller.list_available_tools()
            
            if tools:
                st.subheader("Available MCP Tools")
                
                for tool in tools:
                    with st.expander(f"Tool: {tool['name']}"):
                        st.write(f"**Description:** {tool.get('description', 'No description provided')}")
                        st.write("**Parameters:**")
                        st.json(tool.get('parameters', {}))
                
                # User input for MCP
                st.subheader("Ask a Question")
                mcp_query = st.text_area("Enter a question that might use MCP tools",
                                       "What time is it in Tokyo?")
                
                if st.button("Process with MCP"):
                    if mcp_query:
                        with st.spinner("Processing with MCP..."):
                            try:
                                result = mcp_caller.process_with_tool_calling(mcp_query)
                                
                                st.write("### Response:")
                                st.write(result["response"])
                                
                                if result["tool_calls"]:
                                    with st.expander("Tool Call Details"):
                                        st.write(f"Used {len(result['tool_calls'])} tool calls:")
                                        for i, tool_result in enumerate(result["tool_results"]):
                                            st.write(f"**Tool {i+1}:** {tool_result['name']}")
                                            st.write("Arguments:")
                                            st.json(tool_result['arguments'])
                                            st.write("Result:")
                                            st.json(tool_result['result'])
                            except Exception as e:
                                st.error(f"Error during MCP processing: {str(e)}")
                    else:
                        st.warning("Please enter a question.")
            else:
                st.warning("No tools available from MCP server.")

if __name__ == "__main__":
    main()

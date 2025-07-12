"""
5_MCP.py - Detailed exploration of Model Context Protocol (MCP)

© 2025 NextRun Digital. All Rights Reserved.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import time
import json
import requests
from typing import Dict, Any, List

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from utils.tool_calling import MCPToolCaller
from utils.config import configure_openai, MCP_SERVER_URL, check_mcp_server, get_openai_api_key

def main():
    st.title("Model Context Protocol (MCP)")
    st.markdown("© 2025 NextRun Digital. All Rights Reserved.")
    
    # Check if OpenAI API key is set
    if not configure_openai():
        st.warning("Please set your OpenAI API key in the home page to use this feature.")
        return
    
    # Check MCP server status
    mcp_running = check_mcp_server()
    
    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["What is MCP?", "MCP Explorer", "Building MCP Tools"])
    
    with tab1:
        st.header("Model Context Protocol (MCP)")
        
        st.markdown("""
        ### What is MCP?
        
        **MCP = Model Context Protocol**
        
        MCP is a standard way for LLMs to discover and call tools, defining:
        
        - Tool names
        - Input parameters
        - Expected output formats
        
        ### Why MCP Matters:
        
        - **Standardization**: Avoids ad hoc formats for tool calls
        - **Interoperability**: Makes LLMs work across different apps
        - **Security**: Helps manage security and validation
        - **Orchestration**: Enables complex workflows through agent orchestration
        
        ### How MCP Works:
        
        1. Developer registers tools in MCP with:
           - Name & description
           - Input & output schemas
        
        2. LLM sees tool list → decides which tool to call
        
        3. LLM outputs a structured call (e.g., JSON)
        
        4. The app executes the tool → feeds result back to LLM
        """)
        
        st.subheader("MCP vs. Ad Hoc Tool Calling")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Without MCP")
            st.markdown("""
            - Each application defines its own tool interface
            - LLMs need custom prompting for each application
            - No consistent way to discover available tools
            - Security and validation handled differently per app
            - Difficult to compose tools across applications
            """)
        
        with col2:
            st.markdown("#### With MCP")
            st.markdown("""
            - Standard tool definition format
            - LLMs can discover tools automatically
            - Consistent interface across applications
            - Standardized security and validation
            - Tools can be easily composed across applications
            """)
        
        st.subheader("MCP Architecture")
        st.markdown("""
        MCP consists of several key components:
        
        1. **Tool Registry**: Where tools are registered with metadata and schemas
        
        2. **Discovery Mechanism**: How LLMs find available tools
        
        3. **Execution Engine**: Handles tool invocation and result processing
        
        4. **Security Layer**: Validates and authorizes tool calls
        
        5. **Schema Validation**: Ensures inputs and outputs match expectations
        """)
        
        st.code("""
        # Example of registering a tool with MCP
        mcp_server.register_tool(
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
            function=get_weather_function
        )
        """, language="python")
    
    with tab2:
        st.header("MCP Explorer")
        
        if not mcp_running:
            st.error("MCP server is not running. Please start the MCP server to use this feature.")
            st.info("Run the following command to start the MCP server: `python -m nextrunmcp.main`")
        else:
            # Initialize MCP tool caller
            mcp_caller = MCPToolCaller()
            
            st.success(f"Connected to MCP server at {MCP_SERVER_URL}")
            
            # Get available tools
            tools = mcp_caller.list_available_tools()
            
            if tools:
                st.subheader("Available MCP Tools")
                
                # Create a simple table of tools
                tools_df = pd.DataFrame([{
                    "name": tool["name"],
                    "description": tool.get("description", "No description")
                } for tool in tools])
                
                st.dataframe(tools_df, use_container_width=True)
                
                # Tool details
                st.subheader("Tool Details")
                selected_tool = st.selectbox("Select a tool to explore", 
                                          [tool["name"] for tool in tools])
                
                # Find the selected tool
                tool_details = next((tool for tool in tools if tool["name"] == selected_tool), None)
                
                if tool_details:
                    st.json(tool_details)
                    
                    # Tool testing section
                    st.subheader("Test Tool")
                    
                    # Get parameters from schema
                    if "properties" in tool_details.get("parameters", {}):
                        properties = tool_details["parameters"]["properties"]
                        required = tool_details["parameters"].get("required", [])
                        
                        # Create form for parameters
                        with st.form(f"test_tool_{selected_tool}"):
                            params = {}
                            
                            for param_name, param_details in properties.items():
                                param_type = param_details.get("type", "string")
                                param_desc = param_details.get("description", "")
                                is_required = param_name in required
                                
                                st.write(f"**{param_name}**" + (" (required)" if is_required else ""))
                                st.write(param_desc)
                                
                                if param_type == "string":
                                    params[param_name] = st.text_input(f"Value for {param_name}", key=f"param_{param_name}")
                                elif param_type == "integer":
                                    params[param_name] = st.number_input(f"Value for {param_name}", step=1, key=f"param_{param_name}")
                                elif param_type == "number":
                                    params[param_name] = st.number_input(f"Value for {param_name}", key=f"param_{param_name}")
                                elif param_type == "boolean":
                                    params[param_name] = st.checkbox(f"Value for {param_name}", key=f"param_{param_name}")
                            
                            submit = st.form_submit_button("Call Tool")
                            
                            if submit:
                                # Remove empty optional parameters
                                params = {k: v for k, v in params.items() if v or k in required}
                                
                                # Call the tool
                                with st.spinner(f"Calling {selected_tool}..."):
                                    try:
                                        result = mcp_caller.call_tool(selected_tool, params)
                                        st.subheader("Tool Result:")
                                        st.json(result)
                                    except Exception as e:
                                        st.error(f"Error calling tool: {str(e)}")
                    else:
                        st.info("This tool doesn't have any parameters defined.")
            else:
                st.warning("No tools available from MCP server.")
    
    with tab3:
        st.header("Building MCP Tools")
        
        st.markdown("""
        ### How to Create Your Own MCP Tools
        
        Building MCP-compatible tools involves these steps:
        
        1. **Define Tool Interface**: Create a schema for inputs and outputs
        
        2. **Implement Tool Logic**: Write the function that performs the action
        
        3. **Register with MCP**: Add the tool to the MCP registry
        
        4. **Handle Security**: Implement validation and authorization
        
        5. **Test Integration**: Verify LLM can discover and use the tool
        """)
        
        st.subheader("Example: Creating a Simple MCP Tool")
        
        st.code("""
        # Step 1: Define the tool function
        def calculate_age(birth_date):
            from datetime import datetime
            
            # Parse birth date (format: YYYY-MM-DD)
            birth_date = datetime.strptime(birth_date, "%Y-%m-%d")
            
            # Calculate age
            today = datetime.now()
            age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            
            return {
                "birth_date": birth_date.strftime("%Y-%m-%d"),
                "age": age,
                "calculation_date": today.strftime("%Y-%m-%d")
            }
        
        # Step 2: Register the tool with MCP
        from mcp_server import MCPServer
        
        mcp = MCPServer()
        mcp.register_tool(
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
        
        # Step 3: Start the MCP server
        mcp.start()
        """, language="python")
        
        st.subheader("NextRun MCP Implementation")
        
        st.markdown("""
        The `nextrunmcp` module in this project provides a working MCP server implementation.
        
        To create your own tools:
        
        1. Define tool functions in `nextrunmcp/services.py`
        2. Register them in `nextrunmcp/main.py`
        3. Run the server with `python -m nextrunmcp.main`
        
        Example from the nextrunmcp module:
        """)
        
        st.code("""
        # From nextrunmcp/services.py
        def get_current_time(timezone=None):
            from datetime import datetime
            import pytz
            
            if timezone:
                try:
                    tz = pytz.timezone(timezone)
                    current_time = datetime.now(tz)
                except pytz.exceptions.UnknownTimeZoneError:
                    return {"error": f"Unknown timezone: {timezone}"}
            else:
                current_time = datetime.now()
                
            return {
                "current_time": current_time.strftime("%Y-%m-%d %H:%M:%S"),
                "timezone": timezone or "local"
            }
        
        # From nextrunmcp/main.py
        app.add_api_route(
            "/tools",
            endpoints.list_tools,
            methods=["GET"],
            response_model=List[Tool],
            description="List available tools"
        )
        
        # Register tools
        tools_registry.register_tool(
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
            function=services.get_current_time
        )
        """, language="python")
        
        if mcp_running:
            st.success("The MCP server is running! You can see the available tools in the MCP Explorer tab.")
        else:
            st.warning("To see MCP in action, start the MCP server: `python -m nextrunmcp.main`")

if __name__ == "__main__":
    main()

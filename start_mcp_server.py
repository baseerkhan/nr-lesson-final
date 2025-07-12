"""
start_mcp_server.py - Script to start the MCP server

© 2025 NextRun Digital. All Rights Reserved.
"""

from mcpserver.main import start_server

if __name__ == "__main__":
    print("Starting MCP server on http://127.0.0.1:8000")
    print("Press Ctrl+C to stop the server")
    start_server()

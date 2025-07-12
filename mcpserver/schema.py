"""
schema.py - Schema definitions for MCP server

© 2025 NextRun Digital. All Rights Reserved.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional

class Tool(BaseModel):
    """Schema for a tool definition"""
    name: str
    description: str
    parameters: Dict[str, Any]

class ToolCallRequest(BaseModel):
    """Schema for a tool call request"""
    parameters: Dict[str, Any] = Field(default_factory=dict)

class ToolCallResponse(BaseModel):
    """Schema for a tool call response"""
    result: Any

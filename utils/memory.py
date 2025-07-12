"""
memory.py - Memory management for conversation history and context

© 2025 NextRun Digital. All Rights Reserved.
"""

import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path
from .config import DATA_DIR

class MemorySystem:
    """Implements a simple memory system for storing conversation history and context"""
    
    def __init__(self, memory_file: str = "conversation_memory.xlsx"):
        """Initialize the memory system"""
        self.memory_path = DATA_DIR / memory_file
        self._ensure_data_dir()
        
    def _ensure_data_dir(self):
        """Ensure data directory exists"""
        os.makedirs(DATA_DIR, exist_ok=True)
    
    def add_memory(self, content: str, memory_type: str = "conversation", metadata: Dict[str, Any] = None) -> str:
        """Add a memory to the system"""
        # Generate a memory ID
        memory_id = f"mem_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Prepare data for storage
        data = {
            'id': memory_id,
            'content': content,
            'type': memory_type,
            'metadata': json.dumps(metadata or {}),
            'created_at': datetime.now().isoformat(),
        }
        
        # Load existing data or create new DataFrame
        if os.path.exists(self.memory_path):
            df = pd.read_excel(self.memory_path)
            df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
        else:
            df = pd.DataFrame([data])
        
        # Save to Excel
        df.to_excel(self.memory_path, index=False)
        
        return memory_id
    
    def get_memories(self, memory_type: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get memories from the system, optionally filtered by type"""
        if not os.path.exists(self.memory_path):
            return []
        
        df = pd.read_excel(self.memory_path)
        
        if memory_type:
            df = df[df['type'] == memory_type]
        
        # Sort by creation time, newest first
        df = df.sort_values('created_at', ascending=False)
        
        # Convert to list of dictionaries
        memories = []
        for _, row in df.head(limit).iterrows():
            memories.append({
                'id': row['id'],
                'content': row['content'],
                'type': row['type'],
                'metadata': json.loads(row['metadata']),
                'created_at': row['created_at']
            })
        
        return memories
    
    def clear_memories(self, memory_type: Optional[str] = None) -> None:
        """Clear memories, optionally filtered by type"""
        if not os.path.exists(self.memory_path):
            return
        
        df = pd.read_excel(self.memory_path)
        
        if memory_type:
            # Delete only memories of the specified type
            df = df[df['type'] != memory_type]
            df.to_excel(self.memory_path, index=False)
        else:
            # Delete all memories
            os.remove(self.memory_path)

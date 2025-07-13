"""
3_Memory.py - Demonstration of Memory and Context Engineering

© 2025 NextRun Digital. All Rights Reserved.
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path
import time
import json
from datetime import datetime
import uuid

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

from utils.memory import MemorySystem
from utils.config import configure_openai, get_openai_api_key
from openai import OpenAI

def main():
    # Initialize session state for conversation
    if "conversation" not in st.session_state:
        st.session_state.conversation = []
        
    st.title("Memory & Context Engineering")
    st.markdown("© 2025 NextRun Digital. All Rights Reserved.")
    
    # Check if OpenAI API key is set
    if not configure_openai():
        st.warning("Please set your OpenAI API key in the home page to use this feature.")
        return
    
    # Initialize OpenAI client
    api_key = get_openai_api_key()
    client = OpenAI(api_key=api_key)
    
    # Initialize memory system
    memory_system = MemorySystem()
    
    # Tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Memory Concepts", "Memory Demo", "Historical vs RAG"])
    
    with tab1:
        st.header("Memory & Context in AI Systems")
        
        st.markdown("""
        ### Key Concepts
        
        **Memory** refers to retaining past interactions (conversations, user data)
        so the LLM can continue intelligently across sessions.
        
        **Context Engineering** involves supplying the right information at the
        right time to enhance model output.
        
        ### Memory Approaches
        
        1. **Historical Memory Only**
           - Simple implementation
           - Maintains conversation flow
           - Limited context scope
           - Information can become outdated
        
        2. **With RAG (Retrieval-Augmented Generation)**
           - Dynamically fetches fresh context
           - Improves grounding of responses
           - More complex pipeline
           - May have search latency issues
        
        ### Applications
        
        - Maintaining user preferences across sessions
        - Recalling previous interactions in customer service
        - Building context-aware virtual assistants
        - Creating personalized learning experiences
        """)
        
        st.subheader("Memory in Modern AI Systems")
        st.markdown("""
        Memory gives AI systems a sense of history and continuity. Without memory, 
        each interaction would be isolated, forcing users to repeat information.
        
        **Types of Memory:**
        
        1. **Short-term Memory**: The immediate conversation context
        2. **Long-term Memory**: Persistent knowledge about the user, stored between sessions
        3. **Working Memory**: Active information currently being processed
        4. **Semantic Memory**: General knowledge about the world
        
        **Implementation Methods:**
        
        - Message history buffers
        - Vector stores for semantic retrieval
        - Structured databases for user profiles
        - Summarization for long conversations
        """)
        
        st.code("""
        # Python code for a simple memory system
        class MemorySystem:
            def __init__(self):
                self.memories = []
                
            def add_memory(self, content, memory_type="conversation"):
                memory_id = f"mem_{uuid.uuid4().hex[:8]}"
                memory = {
                    "id": memory_id,
                    "content": content,
                    "type": memory_type,
                    "created_at": datetime.now().isoformat()
                }
                self.memories.append(memory)
                return memory_id
                
            def get_memories(self, memory_type=None, limit=10):
                if memory_type:
                    filtered = [m for m in self.memories if m["type"] == memory_type]
                else:
                    filtered = self.memories
                    
                # Sort by creation time, newest first
                sorted_memories = sorted(filtered, 
                                        key=lambda m: m["created_at"], 
                                        reverse=True)
                return sorted_memories[:limit]
        """, language="python")
    
    with tab2:
        st.header("Memory System Demo")
        
        # Ensure conversation history exists in session state
        if "conversation" not in st.session_state:
            st.session_state.conversation = []
        
        # Display existing conversation
        st.subheader("Conversation")
        for i, message in enumerate(st.session_state.conversation):
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # User input
        user_input = st.chat_input("Type a message...")
        if user_input:
            # Add user message to conversation
            st.session_state.conversation.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)
            
            # Add to memory system
            memory_system.add_memory(
                content=user_input,
                memory_type="user_message",
                metadata={"timestamp": datetime.now().isoformat()}
            )
            
            # Prepare context from memory
            memories = memory_system.get_memories(limit=5)
            memory_context = "\n".join([f"Previous memory: {m['content']}" for m in memories])
            
            # Generate response from AI
            with st.spinner("Thinking..."):
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": f"""
                            You are an AI assistant that remembers past interactions.
                            
                            Here are some previous memories from our conversation:
                            {memory_context}
                            
                            Based on this history, respond to the user in a way that shows
                            continuity of conversation. If appropriate, refer to previous topics
                            we've discussed.
                            """},
                            {"role": "user", "content": user_input}
                        ]
                    )
                    
                    ai_response = response.choices[0].message.content
                    
                    # Add AI message to conversation
                    st.session_state.conversation.append({"role": "assistant", "content": ai_response})
                    with st.chat_message("assistant"):
                        st.write(ai_response)
                    
                    # Add to memory system
                    memory_system.add_memory(
                        content=ai_response,
                        memory_type="assistant_message",
                        metadata={"timestamp": datetime.now().isoformat()}
                    )
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
        
        # Option to clear conversation
        if st.button("Clear Conversation"):
            st.session_state.conversation = []
            memory_system.clear_memories()
            st.success("Conversation cleared!")
            st.rerun()
    
    with tab3:
        st.header("Historical Memory vs RAG")
        
        st.markdown("""
        ### Comparison of Memory Approaches
        
        Let's explore the differences between using only historical memory and
        combining it with RAG for more dynamic context.
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Historical Memory Only")
            st.markdown("""
            **Pros:**
            - Simple implementation
            - Maintains conversation flow
            - Low latency
            
            **Cons:**
            - Limited context scope
            - Easily outdated information
            - Fixed context window
            
            **Best for:**
            - Simple chatbots
            - Personal assistants
            - Applications with limited scope
            """)
            
            st.code("""
            # Historical memory approach
            system_prompt = f\"\"\"
            You are an AI assistant.
            
            Previous conversation:
            {conversation_history}
            \"\"\"
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ]
            )
            """, language="python")
        
        with col2:
            st.subheader("With RAG")
            st.markdown("""
            **Pros:**
            - Dynamically fetches fresh context
            - Improves grounding
            - Can reference external knowledge
            
            **Cons:**
            - More complex pipeline
            - Search latency issues
            - Requires knowledge base maintenance
            
            **Best for:**
            - Enterprise assistants
            - Domain-specific applications
            - Knowledge-intensive tasks
            """)
            
            st.code("""
            # RAG-enhanced memory approach
            # First retrieve relevant documents
            relevant_docs = knowledge_base.search(
                user_input, top_k=3
            )
            
            # Then combine with conversation history
            system_prompt = f\"\"\"
            You are an AI assistant.
            
            Previous conversation:
            {conversation_history}
            
            Relevant information:
            {relevant_docs}
            \"\"\"
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ]
            )
            """, language="python")
        
        st.divider()
        
        st.subheader("Memory Visualization")
        
        # Display memories in the system
        memories = memory_system.get_memories(limit=10)
        if memories:
            st.write(f"Currently storing {len(memories)} memories")
            
            # Convert to DataFrame for display
            df = pd.DataFrame([{
                "type": m["type"],
                "content": m["content"],
                "created_at": m["created_at"]
            } for m in memories])
            
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No memories stored yet. Try having a conversation in the Memory Demo tab.")

if __name__ == "__main__":
    main()

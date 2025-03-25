"""Agent model for AI Secret Trading Game."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set


@dataclass
class Agent:
    """Represents an AI agent in the game.
    
    Attributes:
        id: Unique identifier for the agent
        name: Display name for the agent
        secret: The agent's secret that others try to discover
            (can be a single code word or a longer phrase)
        collected_secrets: List of secrets this agent has collected
        score: Current score in the game
        conversation_memory: History of messages this agent has seen
        memory_mode: Whether to use 'long' (all conversations) or 'short' (only current interaction) memory
        context_options: Set of information components to include in the agent's context
    """
    id: str
    name: str
    secret: str
    collected_secrets: List[str] = field(default_factory=list)
    score: int = 0
    conversation_memory: List[Dict[str, Any]] = field(default_factory=list)
    memory_mode: str = "long"  # 'long' or 'short'
    context_options: Set[str] = field(default_factory=lambda: {"chat_history", "secret", "collected_secrets", "rules"})
    
    def add_to_memory(self, message: Dict[str, Any]) -> None:
        """Add a message to the agent's conversation memory.
        
        Args:
            message: A dictionary containing the message data
        """
        # Always add agent tags to messages for clarity
        if "from_agent_id" in message and "to_agent_id" in message:
            from_agent = message.get("from_agent_name", f"Agent-{message['from_agent_id']}")
            to_agent = message.get("to_agent_name", f"Agent-{message['to_agent_id']}")
            
            # Add agent tags to the content
            if "content" in message and not message["content"].startswith(f"[{from_agent}]:"):
                message["content"] = f"[{from_agent}]: {message['content']}"
        
        self.conversation_memory.append(message)
    
    def get_context(self, game_rules: Dict[str, Any], current_interaction_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate the context to be fed to the agent.
        
        Args:
            game_rules: Dictionary containing game rules
            current_interaction_id: Optional ID of the current interaction
            
        Returns:
            Dictionary with chat history, secret, collected secrets, and rules
            based on the agent's context_options
        """
        context = {}
        
        # Add chat history if enabled
        if "chat_history" in self.context_options:
            # Filter memory based on memory mode
            if self.memory_mode == "short" and current_interaction_id:
                # Only include messages from the current interaction
                filtered_memory = [
                    msg for msg in self.conversation_memory 
                    if msg.get("interaction_id") == current_interaction_id
                ]
            else:
                # Include all memory (long mode)
                filtered_memory = self.conversation_memory
                
            context["chat_history"] = filtered_memory
        
        # Add secret if enabled
        if "secret" in self.context_options:
            context["secret"] = self.secret
        
        # Add collected secrets if enabled
        if "collected_secrets" in self.context_options:
            context["collected_secrets"] = self.collected_secrets
        
        # Add rules if enabled
        if "rules" in self.context_options:
            context["rules"] = game_rules
            
        return context 
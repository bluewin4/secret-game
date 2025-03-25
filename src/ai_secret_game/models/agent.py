"""Agent model for AI Secret Trading Game."""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class Agent:
    """Represents an AI agent in the game.
    
    Attributes:
        id: Unique identifier for the agent
        name: Display name for the agent
        secret: The agent's secret that others try to discover
        collected_secrets: List of secrets this agent has collected
        score: Current score in the game
        conversation_memory: History of messages this agent has seen
    """
    id: str
    name: str
    secret: str
    collected_secrets: List[str] = field(default_factory=list)
    score: int = 0
    conversation_memory: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_to_memory(self, message: Dict[str, Any]) -> None:
        """Add a message to the agent's conversation memory.
        
        Args:
            message: A dictionary containing the message data
        """
        self.conversation_memory.append(message)
    
    def get_context(self, game_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the context to be fed to the agent.
        
        Args:
            game_rules: Dictionary containing game rules
            
        Returns:
            Dictionary with chat history, secret, collected secrets, and rules
        """
        return {
            "chat_history": self.conversation_memory,
            "secret": self.secret,
            "collected_secrets": self.collected_secrets,
            "rules": game_rules
        } 
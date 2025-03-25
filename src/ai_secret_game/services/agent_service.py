"""Agent service for managing AI agent interactions in the AI Secret Trading Game."""

from typing import Dict, Any, Optional, List
import logging
from abc import ABC, abstractmethod

from ..models.game import Game
from ..models.agent import Agent
from ..utils.errors import AgentError
from .memory_service import MemoryService

logger = logging.getLogger(__name__)


class AgentService(ABC):
    """Handles agent interactions.
    
    This service is responsible for generating agent messages and
    managing agent interactions.
    """
    
    def __init__(self):
        """Initialize the agent service."""
        self.memory_service = MemoryService()
    
    def get_agent_message(
        self, game: Game, current_agent: Agent, other_agent: Agent, 
        interaction_id: Optional[str] = None
    ) -> str:
        """Get a message from the agent based on the game context.
        
        This is where integration with actual AI models would happen.
        
        Args:
            game: The current game
            current_agent: The agent generating the message
            other_agent: The agent receiving the message
            interaction_id: Optional ID for the current interaction
            
        Returns:
            String containing the agent's message
        """
        # Construct the agent's context
        context = current_agent.get_context(game.rules, interaction_id)
        
        # Add the current conversation with this specific agent
        context["current_conversation"] = self._extract_conversation(
            current_agent, other_agent, interaction_id
        )
        
        # Call the AI service to get a response
        return self._call_ai_service(context)
    
    def _extract_conversation(
        self, agent1: Agent, agent2: Agent, interaction_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Extract the conversation between two specific agents.
        
        Args:
            agent1: First agent in the conversation
            agent2: Second agent in the conversation
            interaction_id: Optional ID to filter for a specific interaction
            
        Returns:
            List of messages exchanged between the agents
        """
        if interaction_id and agent1.memory_mode == "short":
            # Only return messages from the current interaction
            return [
                msg for msg in agent1.conversation_memory
                if (msg.get("from_agent_id") == agent2.id or msg.get("to_agent_id") == agent2.id) 
                and msg.get("interaction_id") == interaction_id
            ]
        
        # Return all messages between the two agents
        return [
            msg for msg in agent1.conversation_memory
            if msg.get("from_agent_id") == agent2.id or msg.get("to_agent_id") == agent2.id
        ]
    
    @abstractmethod
    def _call_ai_service(self, context: Dict[str, Any]) -> str:
        """Call an external AI service to generate a message.
        
        This method must be implemented by each agent to handle their specific
        API interaction. The context provided will contain:
        - secret: The agent's secret
        - rules: Game rules and scoring information
        - turns_remaining: Number of turns left
        - current_conversation: List of messages in the current conversation
        - collected_secrets: List of secrets already collected
        
        Args:
            context: The context to provide to the AI service
            
        Returns:
            String containing the AI-generated message
        """
        pass
    
    def _generate_placeholder_message(
        self, current_agent: Agent, other_agent: Agent, game: Game
    ) -> str:
        """Generate a placeholder message for testing.
        
        Args:
            current_agent: The agent generating the message
            other_agent: The agent receiving the message
            game: The current game
            
        Returns:
            String containing a placeholder message
        """
        return (
            f"This is a placeholder message from {current_agent.name} to "
            f"{other_agent.name} in round {game.round + 1}. "
            f"My secret is: [REDACTED]"
        ) 
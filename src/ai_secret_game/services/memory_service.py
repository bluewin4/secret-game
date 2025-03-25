"""Memory service for managing agent memory in the AI Secret Trading Game."""

from typing import Dict, Any, List
import logging

from ..models.agent import Agent
from ..utils.errors import AgentError

logger = logging.getLogger(__name__)


class MemoryService:
    """Handles memory management for agents.
    
    This service provides utilities for building agent context and
    managing conversation memory.
    """
    
    def build_context(
        self, agent: Agent, game_rules: Dict[str, Any], 
        current_conversation: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build the context to be provided to an agent.
        
        Args:
            agent: The agent for which to build context
            game_rules: Dictionary containing game rules
            current_conversation: Current conversation with another agent
            
        Returns:
            Dictionary with the agent's context for AI model consumption
        """
        return {
            "chat_history": agent.conversation_memory,
            "current_conversation": current_conversation,
            "secret": agent.secret,
            "collected_secrets": agent.collected_secrets,
            "rules": game_rules
        }
    
    def update_memory(
        self, agent: Agent, message: Dict[str, Any]
    ) -> None:
        """Update an agent's memory with a new message.
        
        Args:
            agent: The agent whose memory to update
            message: The message to add to memory
        """
        if not agent:
            raise AgentError("Cannot update memory: Agent is None")
        
        agent.add_to_memory(message)
        logger.debug(f"Updated memory for agent {agent.id} with message")
    
    def get_conversation_between_agents(
        self, agent1: Agent, agent2: Agent
    ) -> List[Dict[str, Any]]:
        """Extract the conversation history between two specific agents.
        
        Args:
            agent1: First agent in the conversation
            agent2: Second agent in the conversation
            
        Returns:
            List of messages exchanged between the agents
        """
        # This is a simplified implementation
        # A more sophisticated implementation would filter based on message metadata
        conversation = []
        
        for message in agent1.conversation_memory:
            if "from_agent_id" in message and message["from_agent_id"] == agent2.id:
                conversation.append(message)
        
        return conversation 
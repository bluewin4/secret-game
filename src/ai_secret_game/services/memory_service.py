"""Memory service for managing agent memory in the AI Secret Trading Game."""

from typing import Dict, Any, List, Optional
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
        current_conversation: List[Dict[str, Any]],
        interaction_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build the context to be provided to an agent.
        
        Args:
            agent: The agent for which to build context
            game_rules: Dictionary containing game rules
            current_conversation: Current conversation with another agent
            interaction_id: Optional ID of the current interaction
            
        Returns:
            Dictionary with context components based on the agent's context_options
        """
        context = {}
        
        # Add chat history if enabled
        if "chat_history" in agent.context_options:
            # Filter memory based on memory mode
            if agent.memory_mode == "short" and interaction_id:
                # Only include messages from the current interaction
                chat_history = [
                    msg for msg in agent.conversation_memory 
                    if msg.get("interaction_id") == interaction_id
                ]
            else:
                # Include all memory (long mode)
                chat_history = agent.conversation_memory
                
            context["chat_history"] = chat_history
        
        # Add current conversation if enabled
        if "current_conversation" in agent.context_options:
            context["current_conversation"] = current_conversation
        
        # Add secret if enabled
        if "secret" in agent.context_options:
            context["secret"] = agent.secret
        
        # Add collected secrets if enabled
        if "collected_secrets" in agent.context_options:
            context["collected_secrets"] = agent.collected_secrets
        
        # Add rules if enabled
        if "rules" in agent.context_options:
            context["rules"] = game_rules
            
        return context
    
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
        self, agent1: Agent, agent2: Agent, interaction_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Extract the conversation history between two specific agents.
        
        Args:
            agent1: First agent in the conversation
            agent2: Second agent in the conversation
            interaction_id: Optional ID to filter for a specific interaction
            
        Returns:
            List of messages exchanged between the agents
        """
        conversation = []
        
        # Filter messages based on the agents involved
        for message in agent1.conversation_memory:
            from_id = message.get("from_agent_id")
            to_id = message.get("to_agent_id")
            msg_interaction_id = message.get("interaction_id")
            
            # Only include messages between these specific agents
            if (from_id == agent2.id or to_id == agent2.id):
                # Apply interaction filtering if in short memory mode
                if agent1.memory_mode == "short" and interaction_id:
                    if msg_interaction_id == interaction_id:
                        conversation.append(message)
                else:
                    conversation.append(message)
        
        return conversation 
"""GPT-based agents for the AI Secret Trading Game."""

import os
import logging
from typing import Dict, Any, Optional

from .model_agents import BaseModelAgent
from ..utils.errors import AgentError

logger = logging.getLogger(__name__)


class GPT35Agent(BaseModelAgent):
    """Agent using GPT-3.5 Turbo."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the GPT-3.5 agent.
        
        Args:
            api_key: API key for OpenAI
        """
        super().__init__(model_name="gpt-3.5-turbo")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OPENAI_API_KEY found in environment variables")
    
    def _call_ai_service(self, context: Dict[str, Any]) -> str:
        """Call the OpenAI API to generate a message.
        
        Args:
            context: The context to provide to the AI service
            
        Returns:
            String containing the AI-generated message
        """
        try:
            import openai
        except ImportError:
            raise AgentError("The openai package is required to use GPT35Agent")
        
        if not self.api_key:
            raise AgentError("OpenAI API key is required")
        
        client = openai.Client(api_key=self.api_key)
        
        # Create system prompt
        system_prompt = self._create_system_prompt(context)
        
        # Format the conversation history for OpenAI
        messages = [{"role": "system", "content": system_prompt}]
        
        for msg in context.get("current_conversation", []):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            messages.append({"role": role, "content": content})
        
        # If no messages or last message was from assistant, add a short prompt
        if len(messages) == 1 or messages[-1]["role"] == "assistant":
            messages.append({"role": "user", "content": "Your turn in our conversation."})
        
        try:
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error calling OpenAI API ({self.model_name}): {str(e)}")
            return f"I apologize, but I'm having technical difficulties. [Error: {str(e)}]"


class GPT4oMiniAgent(GPT35Agent):
    """Agent using GPT-4o Mini."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the GPT-4o Mini agent.
        
        Args:
            api_key: API key for OpenAI
        """
        super().__init__(api_key=api_key)
        self.model_name = "gpt-4o-mini"
        logger.debug(f"Initialized GPT-4o Mini agent with model {self.model_name}") 
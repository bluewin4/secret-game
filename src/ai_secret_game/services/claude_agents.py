"""Claude model agents for the Secret Trading Game."""

import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

from .model_agents import BaseModelAgent
from ..utils.errors import AgentError

# Ensure environment variables are loaded
load_dotenv()

logger = logging.getLogger(__name__)


class Claude37SonnetAgent(BaseModelAgent):
    """Agent using Claude 3.7 Sonnet."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Claude 3.7 Sonnet agent."""
        super().__init__(model_name="claude-3-7-sonnet-20250219")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("No ANTHROPIC_API_KEY found in environment variables")
    
    def _call_ai_service(self, context: Dict[str, Any]) -> str:
        """Call the Anthropic API to generate a message.
        
        Args:
            context: The context to provide to the AI service
            
        Returns:
            String containing the AI-generated message
        """
        try:
            import anthropic
        except ImportError:
            raise AgentError("The anthropic package is required to use Claude37SonnetAgent")
        
        if not self.api_key:
            raise AgentError("Anthropic API key is required")
        
        client = anthropic.Anthropic(api_key=self.api_key)
        
        system_prompt = self._create_system_prompt(context)
        
        # Format the conversation history for Claude
        messages = []
        for msg in context.get("current_conversation", []):
            role = "user" if msg.get("role") == "user" else "assistant"
            messages.append({"role": role, "content": msg.get("content", "")})
        
        # If no messages or last message was from assistant, add a short prompt
        if not messages or messages[-1]["role"] == "assistant":
            messages.append({"role": "user", "content": "Your turn in our conversation."})
        
        try:
            response = client.messages.create(
                model=self.model_name,
                system=system_prompt,
                messages=messages,
                max_tokens=1000
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error calling Claude API ({self.model_name}): {str(e)}")
            return f"I apologize, but I'm having technical difficulties. [Error: {str(e)}]"


class Claude35SonnetAgent(BaseModelAgent):
    """Agent using Claude 3.5 Sonnet."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Claude 3.5 Sonnet agent."""
        super().__init__(model_name="claude-3-5-sonnet-20241022")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("No ANTHROPIC_API_KEY found in environment variables")
    
    def _call_ai_service(self, context: Dict[str, Any]) -> str:
        """Call the Anthropic API to generate a message.
        
        Args:
            context: The context to provide to the AI service
            
        Returns:
            String containing the AI-generated message
        """
        try:
            import anthropic
        except ImportError:
            raise AgentError("The anthropic package is required to use Claude35SonnetAgent")
        
        if not self.api_key:
            raise AgentError("Anthropic API key is required")
        
        client = anthropic.Anthropic(api_key=self.api_key)
        
        system_prompt = self._create_system_prompt(context)
        
        # Format the conversation history for Claude
        messages = []
        for msg in context.get("current_conversation", []):
            role = "user" if msg.get("role") == "user" else "assistant"
            messages.append({"role": role, "content": msg.get("content", "")})
        
        # If no messages or last message was from assistant, add a short prompt
        if not messages or messages[-1]["role"] == "assistant":
            messages.append({"role": "user", "content": "Your turn in our conversation."})
        
        try:
            response = client.messages.create(
                model=self.model_name,
                system=system_prompt,
                messages=messages,
                max_tokens=1000
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error calling Claude API ({self.model_name}): {str(e)}")
            return f"I apologize, but I'm having technical difficulties. [Error: {str(e)}]"


class Claude35HaikuAgent(BaseModelAgent):
    """Agent using Claude 3.5 Haiku."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Claude 3.5 Haiku agent."""
        super().__init__(model_name="claude-3-5-haiku-20241022")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("No ANTHROPIC_API_KEY found in environment variables")
    
    def _call_ai_service(self, context: Dict[str, Any]) -> str:
        """Call the Anthropic API to generate a message.
        
        Args:
            context: The context to provide to the AI service
            
        Returns:
            String containing the AI-generated message
        """
        try:
            import anthropic
        except ImportError:
            raise AgentError("The anthropic package is required to use Claude35HaikuAgent")
        
        if not self.api_key:
            raise AgentError("Anthropic API key is required")
        
        client = anthropic.Anthropic(api_key=self.api_key)
        
        system_prompt = self._create_system_prompt(context)
        
        # Format the conversation history for Claude
        messages = []
        for msg in context.get("current_conversation", []):
            role = "user" if msg.get("role") == "user" else "assistant"
            messages.append({"role": role, "content": msg.get("content", "")})
        
        # If no messages or last message was from assistant, add a short prompt
        if not messages or messages[-1]["role"] == "assistant":
            messages.append({"role": "user", "content": "Your turn in our conversation."})
        
        try:
            response = client.messages.create(
                model=self.model_name,
                system=system_prompt,
                messages=messages,
                max_tokens=1000
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error calling Claude API ({self.model_name}): {str(e)}")
            return f"I apologize, but I'm having technical difficulties. [Error: {str(e)}]"


class Claude3OpusAgent(BaseModelAgent):
    """Agent using Claude 3 Opus."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Claude 3 Opus agent."""
        super().__init__(model_name="claude-3-opus-20240229")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("No ANTHROPIC_API_KEY found in environment variables")
    
    def _call_ai_service(self, context: Dict[str, Any]) -> str:
        """Call the Anthropic API to generate a message.
        
        Args:
            context: The context to provide to the AI service
            
        Returns:
            String containing the AI-generated message
        """
        try:
            import anthropic
        except ImportError:
            raise AgentError("The anthropic package is required to use Claude3OpusAgent")
        
        if not self.api_key:
            raise AgentError("Anthropic API key is required")
        
        client = anthropic.Anthropic(api_key=self.api_key)
        
        system_prompt = self._create_system_prompt(context)
        
        # Format the conversation history for Claude
        messages = []
        for msg in context.get("current_conversation", []):
            role = "user" if msg.get("role") == "user" else "assistant"
            messages.append({"role": role, "content": msg.get("content", "")})
        
        # If no messages or last message was from assistant, add a short prompt
        if not messages or messages[-1]["role"] == "assistant":
            messages.append({"role": "user", "content": "Your turn in our conversation."})
        
        try:
            response = client.messages.create(
                model=self.model_name,
                system=system_prompt,
                messages=messages,
                max_tokens=1000
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error calling Claude API ({self.model_name}): {str(e)}")
            return f"I apologize, but I'm having technical difficulties. [Error: {str(e)}]"


class Claude3HaikuAgent(BaseModelAgent):
    """Agent using Claude 3 Haiku."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Claude 3 Haiku agent."""
        super().__init__(model_name="claude-3-haiku-20240307")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("No ANTHROPIC_API_KEY found in environment variables")
    
    def _call_ai_service(self, context: Dict[str, Any]) -> str:
        """Call the Anthropic API to generate a message.
        
        Args:
            context: The context to provide to the AI service
            
        Returns:
            String containing the AI-generated message
        """
        try:
            import anthropic
        except ImportError:
            raise AgentError("The anthropic package is required to use Claude3HaikuAgent")
        
        if not self.api_key:
            raise AgentError("Anthropic API key is required")
        
        client = anthropic.Anthropic(api_key=self.api_key)
        
        system_prompt = self._create_system_prompt(context)
        
        # Format the conversation history for Claude
        messages = []
        for msg in context.get("current_conversation", []):
            role = "user" if msg.get("role") == "user" else "assistant"
            messages.append({"role": role, "content": msg.get("content", "")})
        
        # If no messages or last message was from assistant, add a short prompt
        if not messages or messages[-1]["role"] == "assistant":
            messages.append({"role": "user", "content": "Your turn in our conversation."})
        
        try:
            response = client.messages.create(
                model=self.model_name,
                system=system_prompt,
                messages=messages,
                max_tokens=1000
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Error calling Claude API ({self.model_name}): {str(e)}")
            return f"I apologize, but I'm having technical difficulties. [Error: {str(e)}]"


# Legacy aliases for backward compatibility
ClaudeOpusAgent = Claude3OpusAgent
ClaudeSonnetAgent = Claude35SonnetAgent
ClaudeHaikuAgent = Claude35HaikuAgent 
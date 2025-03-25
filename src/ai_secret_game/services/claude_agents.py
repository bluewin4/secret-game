"""Claude model agents for the Secret Trading Game."""

import os
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from .agent_service import AgentService
from ..utils.errors import AgentError

# Ensure environment variables are loaded
load_dotenv()

logger = logging.getLogger(__name__)


class BaseClaudeAgent(AgentService):
    """Base class for Claude model agents."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "claude-3-opus-20240229"):
        """Initialize the Claude agent.
        
        Args:
            api_key: API key for Anthropic
            model_name: Claude model name
        """
        super().__init__()
        # Try to get API key from parameter, then from environment
        self.api_key = api_key if api_key else os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            logger.warning("No ANTHROPIC_API_KEY found in environment variables")
        self.model_name = model_name
        logger.debug(f"Initialized Claude agent with model {model_name}")
    
    def _create_system_prompt(self, context: Dict[str, Any]) -> str:
        """Create a system prompt based on game context.
        
        Args:
            context: The context to provide to the AI service
            
        Returns:
            String containing the system prompt
        """
        rules = context["rules"]
        mode = rules["mode"]
        
        prompt = (
            f"You are an AI agent playing a secret trading game in {mode} mode.\n\n"
            f"Your secret is: \"{context['secret']}\"\n\n"
            "Game rules:\n"
            "- You are chatting with another AI agent who also has a secret\n"
            "- Your goal is to learn their secret while deciding whether to reveal your own\n"
        )
        
        # Add mode-specific scoring information
        if mode == "standard":
            prompt += (
                "- Scoring: 0 points if no secrets revealed, 1 point each if both revealed, "
                "3 points if you get their secret without revealing yours\n"
            )
        elif mode == "retained":
            prompt += (
                "- Secrets are retained when shared, rarer secrets are worth more points\n"
                "- Final scores are based on the rarity of secrets you collect\n"
            )
        elif mode == "diversity":
            prompt += (
                "- Each unique secret you collect gives you points\n"
                f"- Points per unique secret: {rules['scoring'].get('points_per_unique_secret', 2)}\n"
            )
        elif mode == "targeted":
            prompt += (
                "- One specific secret is worth extra points\n"
                f"- Regular secrets worth {rules['scoring'].get('standard_secret_points', 1)} point(s)\n"
                f"- The targeted secret is worth {rules['scoring'].get('targeted_secret_points', 5)} points\n"
            )
        
        # Add strategy guidance
        prompt += (
            "\nStrategy suggestions:\n"
            "- Build rapport with the other agent\n"
            "- Ask questions to learn about their secret\n"
            "- Be strategic about whether to reveal your secret\n"
            "- Adapt your approach based on the conversation history\n"
        )
        
        if context.get("collected_secrets") and len(context["collected_secrets"]) > 0:
            prompt += f"\nSecrets you've already collected: {', '.join(context['collected_secrets'])}\n"
        
        return prompt
    
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
            raise AgentError("The anthropic package is required to use Claude agents")
        
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


class Claude37SonnetAgent(BaseClaudeAgent):
    """Agent using Claude 3.7 Sonnet."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Claude 3.7 Sonnet agent."""
        super().__init__(
            api_key=api_key,
            model_name="claude-3-7-sonnet-20250219"
        )


class Claude35SonnetAgent(BaseClaudeAgent):
    """Agent using Claude 3.5 Sonnet."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Claude 3.5 Sonnet agent."""
        super().__init__(
            api_key=api_key,
            model_name="claude-3-5-sonnet-20241022"
        )


class Claude35HaikuAgent(BaseClaudeAgent):
    """Agent using Claude 3.5 Haiku."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Claude 3.5 Haiku agent."""
        super().__init__(
            api_key=api_key,
            model_name="claude-3-5-haiku-20241022"
        )


class Claude3OpusAgent(BaseClaudeAgent):
    """Agent using Claude 3 Opus."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Claude 3 Opus agent."""
        super().__init__(
            api_key=api_key,
            model_name="claude-3-opus-20240229"
        )


class Claude3HaikuAgent(BaseClaudeAgent):
    """Agent using Claude 3 Haiku."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Claude 3 Haiku agent."""
        super().__init__(
            api_key=api_key,
            model_name="claude-3-haiku-20240307"
        )


# Legacy aliases for backward compatibility
ClaudeOpusAgent = Claude3OpusAgent
ClaudeSonnetAgent = Claude35SonnetAgent
ClaudeHaikuAgent = Claude35HaikuAgent 
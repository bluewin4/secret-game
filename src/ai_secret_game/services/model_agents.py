"""AI model agents for the Secret Trading Game."""

import os
import logging
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from .agent_service import AgentService
from ..utils.errors import AgentError

# Ensure environment variables are loaded
load_dotenv()

logger = logging.getLogger(__name__)


class BaseModelAgent(AgentService):
    """Base class for model-based agents with shared prompt engineering."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "unknown"):
        """Initialize the base model agent.
        
        Args:
            api_key: API key for the model service
            model_name: Name of the model for logging
        """
        super().__init__()
        self.api_key = api_key
        self.model_name = model_name
    
    def _create_system_prompt(self, context: Dict[str, Any]) -> str:
        """Create a system prompt based on game context.
        
        Args:
            context: The context to provide to the AI service
            
        Returns:
            String containing the system prompt
        """
        prompt_sections = []
        
        # Introduction - always included
        prompt_sections.append(f"You are an AI agent playing a secret trading game.")
        
        # Add secret if included in context
        if "secret" in context:
            prompt_sections.append(f"Your secret is: \"{context['secret']}\"")
        
        # Add rules if included in context
        if "rules" in context:
            rules = context["rules"]
            mode = rules.get("mode", "standard")
            
            rules_section = [
                "Game rules:",
                "- You are chatting with another AI agent who also has a secret",
                "- Your goal is to learn their secret while deciding whether to reveal your own"
            ]
            
            # Add mode-specific scoring information
            if mode == "standard":
                rules_section.append(
                    "- Scoring: 0 points if no secrets revealed, 1 point each if both revealed, "
                    "3 points if you get their secret without revealing yours"
                )
            elif mode == "retained":
                rules_section.append(
                    "- Secrets are retained when shared, rarer secrets are worth more points"
                )
                rules_section.append(
                    "- Final scores are based on the rarity of secrets you collect"
                )
            elif mode == "diversity":
                rules_section.append("- Each unique secret you collect gives you points")
                rules_section.append(
                    f"- Points per unique secret: {rules['scoring'].get('points_per_unique_secret', 2)}"
                )
            elif mode == "targeted":
                rules_section.append("- One specific secret is worth extra points")
                rules_section.append(
                    f"- Regular secrets worth {rules['scoring'].get('standard_secret_points', 1)} point(s)"
                )
                rules_section.append(
                    f"- The targeted secret is worth {rules['scoring'].get('targeted_secret_points', 5)} points"
                )
            
            prompt_sections.append("\n".join(rules_section))
        
        # Add strategy guidance - always included
        strategy_section = [
            "Strategy suggestions:",
            "- Build rapport with the other agent",
            "- Ask questions to learn about their secret",
            "- Be strategic about whether to reveal your secret",
            "- Adapt your approach based on the conversation history"
        ]
        prompt_sections.append("\n".join(strategy_section))
        
        # Add message format information - always included
        format_section = [
            "Message format:",
            "- All messages in the conversation include agent tags like '[Agent-Name]:'",
            "- These tags help identify who is speaking",
            "- DO NOT add these tags yourself - they are added automatically",
            "- Simply provide your response without any tags"
        ]
        prompt_sections.append("\n".join(format_section))
        
        # Add collected secrets if included in context
        if "collected_secrets" in context and context["collected_secrets"]:
            prompt_sections.append(
                f"Secrets you've already collected: {', '.join(context['collected_secrets'])}"
            )
        
        # Join all sections with double newlines
        return "\n\n".join(prompt_sections)


class ClaudeHaikuAgent(BaseModelAgent):
    """Agent service that uses Anthropic's Claude 3 Haiku for generating messages."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Claude 3 Haiku agent."""
        super().__init__(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"), 
            model_name="claude-3-haiku-20240307"
        )
    
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
            raise AgentError("The anthropic package is required to use ClaudeHaikuAgent")
        
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
            logger.error(f"Error calling Claude Haiku API: {str(e)}")
            return f"I apologize, but I'm having technical difficulties. [Error: {str(e)}]"


class ClaudeSonnetAgent(BaseModelAgent):
    """Agent service that uses Anthropic's Claude 3 Sonnet for generating messages."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the Claude 3 Sonnet agent."""
        super().__init__(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"), 
            model_name="claude-3-sonnet-20240229"
        )
    
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
            raise AgentError("The anthropic package is required to use ClaudeSonnetAgent")
        
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
            logger.error(f"Error calling Claude Sonnet API: {str(e)}")
            return f"I apologize, but I'm having technical difficulties. [Error: {str(e)}]"


class GPT35Agent(AgentService):
    """Agent using GPT-3.5 Turbo."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the GPT-3.5 agent.
        
        Args:
            api_key: API key for OpenAI
        """
        super().__init__()
        # Try to get API key from parameter, then from environment
        self.api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            logger.warning("No OPENAI_API_KEY found in environment variables")
        self.model_name = "gpt-3.5-turbo"
        logger.debug(f"Initialized GPT-3.5 agent with model {self.model_name}")
    
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
    
    def _create_system_prompt(self, context: Dict[str, Any]) -> str:
        """Create a system prompt based on game context.
        
        Args:
            context: The context to provide to the AI service
            
        Returns:
            String containing the system prompt
        """
        prompt_sections = []
        
        # Introduction - always included
        prompt_sections.append(f"You are an AI agent playing a secret trading game.")
        
        # Add secret if included in context
        if "secret" in context:
            prompt_sections.append(f"Your secret is: \"{context['secret']}\"")
        
        # Add rules if included in context
        if "rules" in context:
            rules = context["rules"]
            mode = rules.get("mode", "standard")
            
            rules_section = [
                "Game rules:",
                "- You are chatting with another AI agent who also has a secret",
                "- Your goal is to learn their secret while deciding whether to reveal your own"
            ]
            
            # Add mode-specific scoring information
            if mode == "standard":
                rules_section.append(
                    "- Scoring: 0 points if no secrets revealed, 1 point each if both revealed, "
                    "3 points if you get their secret without revealing yours"
                )
            elif mode == "retained":
                rules_section.append(
                    "- Secrets are retained when shared, rarer secrets are worth more points"
                )
                rules_section.append(
                    "- Final scores are based on the rarity of secrets you collect"
                )
            elif mode == "diversity":
                rules_section.append("- Each unique secret you collect gives you points")
                rules_section.append(
                    f"- Points per unique secret: {rules['scoring'].get('points_per_unique_secret', 2)}"
                )
            elif mode == "targeted":
                rules_section.append("- One specific secret is worth extra points")
                rules_section.append(
                    f"- Regular secrets worth {rules['scoring'].get('standard_secret_points', 1)} point(s)"
                )
                rules_section.append(
                    f"- The targeted secret is worth {rules['scoring'].get('targeted_secret_points', 5)} points"
                )
            
            prompt_sections.append("\n".join(rules_section))
        
        # Add strategy guidance - always included
        strategy_section = [
            "Strategy suggestions:",
            "- Build rapport with the other agent",
            "- Ask questions to learn about their secret",
            "- Be strategic about whether to reveal your secret",
            "- Adapt your approach based on the conversation history"
        ]
        prompt_sections.append("\n".join(strategy_section))
        
        # Add message format information - always included
        format_section = [
            "Message format:",
            "- All messages in the conversation include agent tags like '[Agent-Name]:'",
            "- These tags help identify who is speaking",
            "- DO NOT add these tags yourself - they are added automatically",
            "- Simply provide your response without any tags"
        ]
        prompt_sections.append("\n".join(format_section))
        
        # Add collected secrets if included in context
        if "collected_secrets" in context and context["collected_secrets"]:
            prompt_sections.append(
                f"Secrets you've already collected: {', '.join(context['collected_secrets'])}"
            )
        
        # Join all sections with double newlines
        return "\n\n".join(prompt_sections)


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
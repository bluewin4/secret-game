# Agent Implementation Guide

This guide explains how to create custom agents for the AI Secret Trading Game. For detailed information about the agent interface and API setup, see [Agent Interface](agent_interface.md) and [API Setup](api_setup.md).

## Overview

The game uses a modular agent system where each agent is responsible for:
1. Generating messages based on game context
2. Managing its own memory and state
3. Interacting with an AI service (e.g., OpenAI, Anthropic)

## Base Classes

### AgentService (Abstract Base Class)
Located in `src/ai_secret_game/services/agent_service.py`, this is the root interface that defines the core agent functionality:

```python
class AgentService(ABC):
    def get_agent_message(self, game: Game, current_agent: Agent, other_agent: Agent, interaction_id: Optional[str] = None) -> str:
        """Get a message from the agent based on the game context."""
        pass

    @abstractmethod
    def _call_ai_service(self, context: Dict[str, Any]) -> str:
        """Call an external AI service to generate a message."""
        pass
```

### BaseModelAgent
Located in `src/ai_secret_game/services/model_agents.py`, this class provides common functionality for AI model-based agents:

```python
class BaseModelAgent(AgentService):
    def __init__(self, model_name: str):
        """Initialize the agent with a model name."""
        self.model_name = model_name
        logger.debug(f"Initialized {self.__class__.__name__} with model {model_name}")

    def _create_system_prompt(self, context: Dict[str, Any]) -> str:
        """Create a system prompt based on game context.
        
        The prompt includes:
        1. Introduction and role
        2. Agent's secret (if provided)
        3. Game rules and scoring information
        4. Turns remaining
        5. Message format instructions
        6. Collected secrets (if any)
        """
        pass

    @abstractmethod
    def _call_ai_service(self, context: Dict[str, Any]) -> str:
        """Call an external AI service to generate a message."""
        pass
```

## Context Structure

The context provided to agents includes various components that can be selectively enabled through context options. See [Agent Interface](agent_interface.md#context-options) for details on configuring which components are included.

The full context structure includes:

```python
{
    # Only included if "chat_history" is in context_options
    "chat_history": [
        {
            "role": "user" | "assistant",
            "content": "[Agent-Name]: Message content",
            "from_agent_id": "ID of the sender",
            "from_agent_name": "Name of the sender",
            "to_agent_id": "ID of the recipient",
            "to_agent_name": "Name of the recipient",
            "interaction_id": "ID of the interaction"
        }
    ],
    
    # Only included if "current_conversation" is in context_options
    "current_conversation": [
        # Messages exchanged with the current conversation partner
    ],
    
    # Only included if "secret" is in context_options
    "secret": "The agent's secret",
    
    # Only included if "collected_secrets" is in context_options
    "collected_secrets": [
        # List of secrets this agent has collected
    ],
    
    # Only included if "rules" is in context_options
    "rules": {
        "mode": "standard" | "retained" | "diversity" | "targeted",
        "scoring": {
            # Mode-specific scoring rules
        },
        "max_rounds": 10,
        "messages_per_round": 3,
        "current_round": 1,
        # Optional fields for targeted mode
        "targeted_secret": "High-value secret",
        "targeted_secret_points": 5
    },
    
    # Only included if "rules" is in context_options
    "turns_remaining": int  # Number of turns left in the game
}
```

## Example Implementation

Here's an example of implementing a custom agent using a hypothetical AI service:

```python
from typing import Dict, Any, Optional
import os
import logging
from dotenv import load_dotenv

from .model_agents import BaseModelAgent
from ..utils.errors import AgentError

logger = logging.getLogger(__name__)

class CustomAgent(BaseModelAgent):
    """Custom agent using a hypothetical AI service."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize the custom agent."""
        super().__init__(model_name="custom-model")
        self.api_key = api_key or os.getenv("CUSTOM_API_KEY")
        if not self.api_key:
            logger.warning("No CUSTOM_API_KEY found in environment variables")
    
    def _call_ai_service(self, context: Dict[str, Any]) -> str:
        """Call the custom AI service to generate a message."""
        try:
            # Import your AI service client
            import custom_ai_client
            
            if not self.api_key:
                raise AgentError("Custom API key is required")
            
            # Initialize your client
            client = custom_ai_client.Client(api_key=self.api_key)
            
            # Create system prompt using the base class method
            system_prompt = self._create_system_prompt(context)
            
            # Format messages for your service
            messages = [
                {"role": "system", "content": system_prompt}
            ]
            
            # Add conversation history
            for msg in context.get("current_conversation", []):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                messages.append({"role": role, "content": content})
            
            # Add turn prompt if needed
            if len(messages) == 1 or messages[-1]["role"] == "assistant":
                messages.append({"role": "user", "content": "Your turn in our conversation."})
            
            # Call your service
            response = client.generate(
                model=self.model_name,
                messages=messages,
                max_tokens=1000
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error calling Custom API ({self.model_name}): {str(e)}")
            return f"I apologize, but I'm having technical difficulties. [Error: {str(e)}]"
```

## Best Practices

1. **Inheritance**
   - Always inherit from `BaseModelAgent` for AI model-based agents
   - Implement only the necessary methods
   - Use `super()` calls appropriately
   - Let the base class handle prompt construction

2. **Error Handling**
   - Catch and log all exceptions
   - Provide meaningful error messages
   - Return graceful fallback responses
   - See [API Setup](api_setup.md#troubleshooting) for common error patterns

3. **Configuration**
   - Use environment variables for API keys
   - Provide fallback values where appropriate
   - Log configuration issues
   - Follow the setup instructions in [API Setup](api_setup.md#environment-variables)

4. **Memory Management**
   - Use the provided `MemoryService` for context building
   - Respect the agent's memory mode (short/long)
   - Handle conversation history appropriately
   - See [Agent Interface](agent_interface.md#memory-modes) for memory mode details

5. **Prompt Construction**
   - Use the base class's `_create_system_prompt` method
   - The base class handles all standard prompt sections
   - Format messages according to your AI service's requirements
   - Follow the prompt engineering tips in [Agent Interface](agent_interface.md#prompt-engineering-tips)

6. **Logging**
   - Use the provided logger
   - Log important events and errors
   - Include relevant context in log messages

## Testing

When implementing a custom agent, you should:

1. Test initialization with and without API keys
2. Test message generation with various contexts
3. Test error handling and fallback responses
4. Test memory management and context building
5. Test prompt construction and formatting

Example test:

```python
import pytest
from unittest.mock import patch, MagicMock

def test_custom_agent_initialization():
    agent = CustomAgent()
    assert agent.model_name == "custom-model"
    assert agent.api_key is None

def test_custom_agent_with_api_key():
    agent = CustomAgent(api_key="test-key")
    assert agent.api_key == "test-key"

@patch('custom_ai_client.Client')
def test_custom_agent_message_generation(mock_client):
    agent = CustomAgent(api_key="test-key")
    mock_response = MagicMock()
    mock_response.text = "Test response"
    mock_client.return_value.generate.return_value = mock_response
    
    context = {
        "secret": "test_secret",
        "rules": {"mode": "standard"},
        "current_conversation": []
    }
    
    response = agent._call_ai_service(context)
    assert response == "Test response"
```

## Integration

To use your custom agent with the game:

```python
from ai_secret_game.models.agent import Agent
from ai_secret_game.models.game import Game, GameMode
from ai_secret_game.services.game_service import GameService
from custom_agents import CustomAgent

# Create agents
agents = [
    Agent(id="1", name="Agent1", secret="Secret1"),
    Agent(id="2", name="Agent2", secret="Secret2"),
]

# Create your custom agent service
custom_service = CustomAgent()

# Create game service with your agent service
game_service = GameService(agent_service=custom_service)

# Create and run a game
game = game_service.create_game(
    agents=agents,
    mode=GameMode.STANDARD,
    max_rounds=3,
    messages_per_round=2
)

results = game_service.run_game(game)
```

## Common Pitfalls

1. **Memory Management**
   - Don't modify the agent's memory directly
   - Use the provided memory service methods
   - Handle conversation history properly

2. **Error Handling**
   - Always catch and log exceptions
   - Provide meaningful error messages
   - Return graceful fallback responses

3. **Context Building**
   - Don't modify the context dictionary
   - Use the provided context options
   - Include all necessary information

4. **API Integration**
   - Handle API errors gracefully
   - Use proper authentication
   - Follow rate limits and quotas 
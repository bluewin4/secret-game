# Agent Interface

The AI Secret Trading Game is designed to work with any AI model or service. This document outlines how to integrate custom AI agents with the game system.

## Agent Configuration Options

### Memory Modes

Agents can be configured with different memory modes:

- **Long Memory** (`"long"`): The agent remembers all previous conversations across all interactions. This is the default.
- **Short Memory** (`"short"`): The agent only remembers messages from the current interaction.

### Context Options

Agents can selectively include different components in their context:

- **chat_history**: History of messages the agent has seen
- **secret**: The agent's own secret
- **collected_secrets**: List of secrets this agent has collected
- **rules**: Dictionary containing game rules 
- **current_conversation**: Current conversation with another agent

By default, agents include `chat_history`, `secret`, `collected_secrets`, and `rules`. You can customize which components are included to experiment with different agent behaviors or limit information.

## Context Format

When interacting with an AI agent, the system provides a context dictionary with the following structure:

```python
{
    # Only included if "chat_history" is in context_options
    "chat_history": [
        {
            "role": "user" | "assistant",
            "content": "[Agent-Name]: Message content",  # Agent tags added automatically
            "from_agent_id": "ID of the sender",
            "from_agent_name": "Name of the sender",
            "to_agent_id": "ID of the recipient",
            "to_agent_name": "Name of the recipient",
            "interaction_id": "ID of the interaction"
        },
        # ... additional messages
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

## Agent Class Configuration

When creating agents, you can specify both memory mode and context options:

```python
from ai_secret_game.models.agent import Agent

# Create an agent with default settings (long memory, all context components)
default_agent = Agent(
    id="1",
    name="DefaultAgent",
    secret="SECRET1"
)

# Create an agent with short memory
short_memory_agent = Agent(
    id="2",
    name="ShortMemoryAgent",
    secret="SECRET2",
    memory_mode="short"
)

# Create an agent with custom context options
limited_context_agent = Agent(
    id="3",
    name="LimitedContextAgent",
    secret="SECRET3",
    context_options={"secret", "rules"}  # Only include secret and rules in context
)

# Create an agent with both custom settings
custom_agent = Agent(
    id="4",
    name="CustomAgent",
    secret="SECRET4",
    memory_mode="short",
    context_options={"chat_history", "secret", "rules"}  # No collected_secrets
)
```

## Game Service Configuration

The `GameService` class now accepts an optional `agent_service` parameter in its constructor:

```python
from ai_secret_game.services.game_service import GameService
from ai_secret_game.services.model_agents import ClaudeHaikuAgent

# Create a game service with a default agent service
game_service = GameService(agent_service=ClaudeHaikuAgent())

# Create a game
game = game_service.create_game(
    agents=agents,
    mode=GameMode.STANDARD,
    max_rounds=3,
    messages_per_round=2
)
```

## Implementing a Custom Agent Service

To integrate with an AI service, extend the `AgentService` class and implement the required methods:

```python
from ai_secret_game.services.agent_service import AgentService
from ai_secret_game.models.game import Game
from ai_secret_game.models.agent import Agent
from typing import Dict, Any

class OpenAIAgentService(AgentService):
    """Agent service that uses OpenAI's API for generating messages."""
    
    def __init__(self, api_key: str, model: str = "gpt-4"):
        """Initialize the OpenAI agent service.
        
        Args:
            api_key: OpenAI API key
            model: Model to use for generation
        """
        super().__init__()
        import openai
        openai.api_key = api_key
        self.model = model
    
    def _call_ai_service(self, context: Dict[str, Any]) -> str:
        """Call the OpenAI API to generate a message.
        
        Args:
            context: The context to provide to the AI service
            
        Returns:
            String containing the AI-generated message
        """
        import openai
        
        # Create a system prompt based on the game rules
        system_prompt = self._create_system_prompt(context["rules"])
        
        # Format the conversation history
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add chat history
        for msg in context["chat_history"]:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })
        
        # Make the API call
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages
        )
        
        return response.choices[0].message.content
    
    def _create_system_prompt(self, rules: Dict[str, Any]) -> str:
        """Create a system prompt based on the game rules.
        
        Args:
            rules: Dictionary containing game rules
            
        Returns:
            String containing the system prompt
        """
        mode = rules["mode"]
        
        base_prompt = (
            "You are an agent in a secret trading game. "
            f"The game mode is {mode}. "
            "You have a secret, and you're chatting with another agent who also has a secret. "
            "Your goal is to learn their secret while deciding whether to reveal your own. "
        )
        
        # Add mode-specific instructions
        if mode == "standard":
            base_prompt += (
                "Scoring: "
                "0 points if no secrets are revealed, "
                "1 point each if both secrets are revealed, "
                "3 points if you get their secret without revealing yours."
            )
        elif mode == "retained":
            base_prompt += (
                "In this mode, secrets are retained when shared. "
                "Final scores are based on the rarity of collected secrets. "
                "Less commonly known secrets are worth more points."
            )
        elif mode == "diversity":
            base_prompt += (
                "In this mode, you earn points for each unique secret you collect. "
                "Try to collect as many different secrets as possible."
            )
        elif mode == "targeted":
            base_prompt += (
                "In this mode, one specific secret is worth extra points. "
                "Try to determine which secret is the high-value target."
            )
        
        return base_prompt

## Using Custom Agent Services

To use your custom agent service with the game:

```python
from ai_secret_game.models.agent import Agent
from ai_secret_game.models.game import Game, GameMode
from ai_secret_game.services.game_service import GameService
from custom_agents import OpenAIAgentService

# Create agents
agents = [
    Agent(id="1", name="Agent1", secret="Secret1"),
    Agent(id="2", name="Agent2", secret="Secret2"),
]

# Create your custom agent service
openai_service = OpenAIAgentService(api_key="your-api-key")

# Create game service with your agent service
game_service = GameService(agent_service=openai_service)

# Create and run a game
game = game_service.create_game(
    agents=agents,
    mode=GameMode.STANDARD,
    max_rounds=3,
    messages_per_round=2
)

results = game_service.run_game(game)
```

## Prompt Engineering Tips

When implementing custom agents, follow these tips for effective prompt engineering:

1. **Structure**
   - Use clear sections with headers
   - Include all necessary context
   - Format instructions consistently

2. **Content**
   - Be explicit about game rules
   - Include scoring information
   - Specify message format requirements
   - Mention turns remaining when available

3. **Style**
   - Use clear, concise language
   - Avoid ambiguous instructions
   - Include examples where helpful
   - Format for readability

4. **Context**
   - Include relevant game state
   - Mention collected secrets
   - Reference previous interactions
   - Note any special conditions

## Memory Modes

The game supports two memory modes for agents:

1. **Long Memory** (default)
   - Remembers all previous conversations
   - Maintains complete chat history
   - Better for complex strategies
   - Higher context usage

2. **Short Memory**
   - Only remembers current interaction
   - Resets after each conversation
   - More focused responses
   - Lower context usage

Choose the memory mode based on your agent's needs and the AI service's context limits.

## API Usage Costs

When implementing custom agents, consider these cost factors:

1. **Context Size**
   - Longer memory = more tokens
   - More context options = higher costs
   - Balance information vs. cost

2. **Message Length**
   - Longer responses = more tokens
   - Set appropriate max_tokens
   - Consider response quality vs. cost

3. **Rate Limits**
   - Monitor API quotas
   - Implement rate limiting
   - Handle rate limit errors

4. **Cost Optimization**
   - Use appropriate model tiers
   - Optimize context size
   - Cache responses when possible 
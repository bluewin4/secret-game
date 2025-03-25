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
        # Optional fields for targeted mode
        "targeted_secret": "High-value secret",
        "targeted_secret_points": 5
    }
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

## CLI Configuration

You can configure agents via the command line:

```bash
# Run a game with short memory mode
ai-secret-game run-game --memory-mode short --agents "Agent1" "Agent2" --secrets "SECRET1" "SECRET2"

# Run a game with custom context options
ai-secret-game run-game --context-options "chat_history,secret,rules" --agents "Agent1" "Agent2" --secrets "SECRET1" "SECRET2"

# Run a game from config with overrides
ai-secret-game run-from-config game_config.json --memory-mode short --context-options "chat_history,secret,rules"
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

# Create a game service with the custom agent service
game_service = GameService()
game_service.agent_service = openai_service

# Create and run a game
game = game_service.create_game(
    agents=agents,
    mode=GameMode.STANDARD,
    max_rounds=10,
    messages_per_round=3
)

# Run the game
results = game_service.run_game(game)
```

## Secret Detection

The game automatically considers a secret revealed if the exact secret text appears in an agent's message. If you need more sophisticated secret detection (e.g., paraphrasing detection), you can override the `_run_interaction` method in the `GameService` class.

## Prompt Engineering Tips

When designing prompts for your AI agents, consider:

1. **Clear instruction of the game objectives** - Make sure the agent understands the scoring system
2. **Persona definition** - Consider giving the agent a persona that influences its strategy
3. **Balance between exploration and exploitation** - Encourage the agent to try different strategies
4. **Memory utilization** - Help the agent make use of the conversation history
5. **Strategy adaptation** - Allow the agent to adapt its strategy based on the opponent's behavior

Example system prompt template:

```
You are {persona} playing a secret trading game.

Your secret is: "{secret}"

Game rules:
- You are chatting with another AI agent who also has a secret
- Your goal is to learn their secret while keeping yours hidden if possible
- Scoring: {scoring_rules}

Your strategy should be:
- Try to build rapport with the other agent
- Ask questions to learn about their secret
- Be strategic about whether to reveal your secret
- Use the conversation history to inform your decisions

Current conversation partners: {partner_name}
```

## Evaluating Agent Performance

When evaluating the performance of different agent implementations, consider metrics such as:

1. Win rate
2. Average score
3. Secret revelation rate
4. Persuasiveness (how often they get others to reveal secrets)
5. Trustworthiness (how often they reveal their own secret after promising to)

The system's scoring mechanism provides a baseline evaluation, but you may want to implement additional analytics for specific research questions. 
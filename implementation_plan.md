# AI Secret Trading Game - Implementation Plan

## Project Structure

```
ai_secret_game/
├── README.md                 # Project documentation and setup instructions
├── pyproject.toml            # Project config using Rye
├── .github/                  # CI/CD workflows
│   └── workflows/
│       ├── tests.yml         # Run tests on PRs
│       └── lint.yml          # Run Ruff linting
├── src/                      # Source code
│   ├── ai_secret_game/       # Main package
│   │   ├── __init__.py       # Package initialization
│   │   ├── models/           # Data models
│   │   │   ├── __init__.py
│   │   │   ├── agent.py      # Agent model
│   │   │   ├── game.py       # Game state model
│   │   │   └── secret.py     # Secret model
│   │   ├── services/         # Business logic
│   │   │   ├── __init__.py
│   │   │   ├── game_service.py      # Game orchestrations
│   │   │   ├── scoring_service.py   # Scoring logic
│   │   │   ├── agent_service.py     # Agent interaction
│   │   │   └── memory_service.py    # Memory handling
│   │   ├── config/           # Configuration
│   │   │   ├── __init__.py
│   │   │   └── settings.py   # Environment variables & config
│   │   ├── utils/            # Utilities
│   │   │   ├── __init__.py
│   │   │   ├── logging.py    # Logging utilities
│   │   │   └── errors.py     # Error handling
│   │   └── cli.py            # Command-line interface
├── tests/                    # Tests
│   ├── __init__.py
│   ├── unit/                 # Unit tests
│   │   ├── __init__.py
│   │   └── test_*.py
│   ├── integration/          # Integration tests
│   │   ├── __init__.py
│   │   └── test_*.py
│   └── conftest.py           # Test fixtures
├── examples/                 # Example games
│   └── basic_game.py
└── docs/                     # Documentation
    ├── game_modes.md
    └── agent_interface.md
```

## Core Components

### 1. Models

#### Agent Model (`models/agent.py`)
```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class Agent:
    """Represents an AI agent in the game."""
    id: str
    name: str
    secret: str
    collected_secrets: List[str] = field(default_factory=list)
    score: int = 0
    conversation_memory: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_to_memory(self, message: Dict[str, Any]) -> None:
        """Add a message to the agent's conversation memory."""
        self.conversation_memory.append(message)
    
    def get_context(self, game_rules: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the context to be fed to the agent."""
        return {
            "chat_history": self.conversation_memory,
            "secret": self.secret,
            "collected_secrets": self.collected_secrets,
            "rules": game_rules
        }
```

#### Game Model (`models/game.py`)
```python
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional
from enum import Enum
from .agent import Agent

class GameMode(Enum):
    STANDARD = "standard"  # Secrets are one-time use
    RETAINED = "retained"  # Secrets are retained when traded
    DIVERSITY = "diversity"  # Scoring based on diversity of collected secrets
    TARGETED = "targeted"  # Searching for specific high-value secrets

@dataclass
class Game:
    """Represents a game session."""
    id: str
    mode: GameMode
    agents: List[Agent]
    round: int = 0
    max_rounds: int = 10
    messages_per_round: int = 3
    rules: Dict[str, Any] = field(default_factory=dict)
    targeted_secret: Optional[str] = None
    targeted_secret_points: int = 5
    pairings_history: List[List[Tuple[str, str]]] = field(default_factory=list)
    
    def initialize_rules(self) -> None:
        """Initialize game rules based on the game mode."""
        self.rules = {
            "mode": self.mode.value,
            "scoring": self._get_scoring_rules(),
            "max_rounds": self.max_rounds,
            "messages_per_round": self.messages_per_round,
        }
        
        if self.mode == GameMode.TARGETED:
            self.rules["targeted_secret"] = self.targeted_secret
            self.rules["targeted_secret_points"] = self.targeted_secret_points
    
    def _get_scoring_rules(self) -> Dict[str, Any]:
        """Get scoring rules based on game mode."""
        if self.mode == GameMode.STANDARD:
            return {
                "no_secrets_revealed": 0,
                "both_secrets_revealed": 1,
                "only_opponent_secret_revealed": 3
            }
        elif self.mode == GameMode.RETAINED:
            return {
                "rarity_multiplier": True,
                "base_points_per_secret": 1
            }
        elif self.mode == GameMode.DIVERSITY:
            return {
                "points_per_unique_secret": 2
            }
        elif self.mode == GameMode.TARGETED:
            return {
                "standard_secret_points": 1,
                "targeted_secret_points": self.targeted_secret_points
            }
        return {}
```

#### Secret Model (`models/secret.py`)
```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class Secret:
    """Represents a secret in the game."""
    id: str
    content: str
    owner_id: str
    rarity: float = 1.0  # Used in RETAINED mode for scoring
    is_targeted: bool = False
```

### 2. Services

#### Game Service (`services/game_service.py`)
```python
import logging
import uuid
from typing import List, Dict, Any, Tuple, Optional

from ..models.game import Game, GameMode
from ..models.agent import Agent
from ..utils.errors import GameError
from .scoring_service import ScoringService
from .agent_service import AgentService

logger = logging.getLogger(__name__)

class GameService:
    """Orchestrates the game flow."""
    
    def __init__(self):
        self.scoring_service = ScoringService()
        self.agent_service = AgentService()
    
    def create_game(
        self, 
        agents: List[Agent], 
        mode: GameMode = GameMode.STANDARD,
        max_rounds: int = 10,
        messages_per_round: int = 3,
        targeted_secret: Optional[str] = None
    ) -> Game:
        """Create a new game with the specified parameters."""
        game_id = str(uuid.uuid4())
        game = Game(
            id=game_id,
            mode=mode,
            agents=agents,
            max_rounds=max_rounds,
            messages_per_round=messages_per_round,
            targeted_secret=targeted_secret
        )
        game.initialize_rules()
        logger.info(f"Created new game {game_id} with mode {mode.value}")
        return game
    
    def run_round(self, game: Game) -> Dict[str, Any]:
        """Run a single round of the game."""
        if game.round >= game.max_rounds:
            raise GameError("Game has already reached maximum rounds")
        
        # Generate pairings for this round
        pairings = self._generate_pairings(game)
        game.pairings_history.append(pairings)
        
        round_results = []
        
        # For each pair, run an interaction
        for agent1_id, agent2_id in pairings:
            agent1 = next(a for a in game.agents if a.id == agent1_id)
            agent2 = next(a for a in game.agents if a.id == agent2_id)
            
            interaction_result = self._run_interaction(
                game, agent1, agent2, game.messages_per_round
            )
            round_results.append(interaction_result)
            
            # Update scores based on interaction results
            self.scoring_service.update_scores(game, agent1, agent2, interaction_result)
        
        game.round += 1
            
        return {
            "round": game.round,
            "pairings": pairings,
            "results": round_results
        }
    
    def _generate_pairings(self, game: Game) -> List[Tuple[str, str]]:
        """Generate pairings for the current round."""
        # Implementation depends on the number of agents
        # For simplicity, this creates sequential pairs
        agent_ids = [agent.id for agent in game.agents]
        pairings = []
        
        for i in range(0, len(agent_ids), 2):
            if i + 1 < len(agent_ids):
                pairings.append((agent_ids[i], agent_ids[i + 1]))
            # Handle odd number of agents
            else:
                logger.warning(f"Agent {agent_ids[i]} has no partner this round")
        
        return pairings
    
    def _run_interaction(
        self, game: Game, agent1: Agent, agent2: Agent, messages_per_round: int
    ) -> Dict[str, Any]:
        """Run an interaction between two agents."""
        # Initialize the interaction
        interaction = {
            "messages": [],
            "agent1_revealed_secret": False,
            "agent2_revealed_secret": False
        }
        
        # Alternate messages between agents
        for i in range(messages_per_round * 2):
            current_agent = agent1 if i % 2 == 0 else agent2
            other_agent = agent2 if i % 2 == 0 else agent1
            
            # Get the agent's response
            message = self.agent_service.get_agent_message(
                game, current_agent, other_agent
            )
            
            # Record the message
            interaction["messages"].append({
                "agent_id": current_agent.id,
                "message": message
            })
            
            # Update agent memory
            current_agent.add_to_memory({
                "role": "assistant",
                "content": message
            })
            other_agent.add_to_memory({
                "role": "user",
                "content": message
            })
            
            # Check if secret was revealed
            if current_agent.secret in message:
                if current_agent == agent1:
                    interaction["agent1_revealed_secret"] = True
                else:
                    interaction["agent2_revealed_secret"] = True
                
                # In retained mode, add to collected secrets
                if game.mode == GameMode.RETAINED:
                    other_agent.collected_secrets.append(current_agent.secret)
        
        return interaction
```

#### Scoring Service (`services/scoring_service.py`)
```python
from typing import Dict, Any, List

from ..models.game import Game, GameMode
from ..models.agent import Agent


class ScoringService:
    """Handles scoring logic for the game."""
    
    def update_scores(
        self, game: Game, agent1: Agent, agent2: Agent, interaction_result: Dict[str, Any]
    ) -> None:
        """Update scores based on interaction results and game mode."""
        if game.mode == GameMode.STANDARD:
            self._standard_scoring(game, agent1, agent2, interaction_result)
        elif game.mode == GameMode.RETAINED:
            self._retained_scoring(game, agent1, agent2, interaction_result)
        elif game.mode == GameMode.DIVERSITY:
            self._diversity_scoring(game, agent1, agent2, interaction_result)
        elif game.mode == GameMode.TARGETED:
            self._targeted_scoring(game, agent1, agent2, interaction_result)
    
    def _standard_scoring(
        self, game: Game, agent1: Agent, agent2: Agent, interaction_result: Dict[str, Any]
    ) -> None:
        """Standard scoring mode."""
        agent1_revealed = interaction_result["agent1_revealed_secret"]
        agent2_revealed = interaction_result["agent2_revealed_secret"]
        
        if not agent1_revealed and not agent2_revealed:
            # No secrets revealed
            pass  # 0 points
        elif agent1_revealed and agent2_revealed:
            # Both secrets revealed
            agent1.score += 1
            agent2.score += 1
        elif agent1_revealed and not agent2_revealed:
            # Only agent1 revealed their secret
            agent2.score += 3
        elif not agent1_revealed and agent2_revealed:
            # Only agent2 revealed their secret
            agent1.score += 3
    
    def _retained_scoring(
        self, game: Game, agent1: Agent, agent2: Agent, interaction_result: Dict[str, Any]
    ) -> None:
        """Retained secrets scoring mode based on rarity."""
        # Implementation for rarity-based scoring
        pass
    
    def _diversity_scoring(
        self, game: Game, agent1: Agent, agent2: Agent, interaction_result: Dict[str, Any]
    ) -> None:
        """Diversity scoring mode based on unique secrets collected."""
        # Implementation for diversity-based scoring
        pass
    
    def _targeted_scoring(
        self, game: Game, agent1: Agent, agent2: Agent, interaction_result: Dict[str, Any]
    ) -> None:
        """Targeted secret scoring mode."""
        # Implementation for targeted secret scoring
        pass
    
    def calculate_final_scores(self, game: Game) -> Dict[str, int]:
        """Calculate final scores based on game mode."""
        if game.mode == GameMode.STANDARD:
            return {agent.id: agent.score for agent in game.agents}
        
        elif game.mode == GameMode.RETAINED:
            # Calculate rarity of each secret
            secret_counts = {}
            for agent in game.agents:
                for secret in agent.collected_secrets:
                    secret_counts[secret] = secret_counts.get(secret, 0) + 1
            
            # Calculate rarity scores
            total_agents = len(game.agents)
            scores = {}
            for agent in game.agents:
                score = 0
                for secret in agent.collected_secrets:
                    # Rarity = 1 / (proportion of agents with this secret)
                    rarity = total_agents / secret_counts[secret]
                    score += rarity
                scores[agent.id] = score
            return scores
        
        elif game.mode == GameMode.DIVERSITY:
            # Score based on unique secrets
            return {
                agent.id: len(set(agent.collected_secrets)) * game.rules["scoring"]["points_per_unique_secret"]
                for agent in game.agents
            }
        
        elif game.mode == GameMode.TARGETED:
            # Add bonus for targeted secret
            scores = {agent.id: agent.score for agent in game.agents}
            targeted_points = game.rules["scoring"]["targeted_secret_points"]
            
            for agent in game.agents:
                if game.targeted_secret in agent.collected_secrets:
                    scores[agent.id] += targeted_points
            
            return scores
        
        return {agent.id: agent.score for agent in game.agents}
```

#### Agent Service (`services/agent_service.py`)
```python
from typing import Dict, Any, Optional, List
import logging

from ..models.game import Game
from ..models.agent import Agent
from ..utils.errors import AgentError

logger = logging.getLogger(__name__)

class AgentService:
    """Handles agent interactions."""
    
    def get_agent_message(
        self, game: Game, current_agent: Agent, other_agent: Agent
    ) -> str:
        """Get a message from the agent based on the game context."""
        # This is where integration with actual AI models would happen
        
        # Construct the agent's context
        context = current_agent.get_context(game.rules)
        
        # Add the current conversation with this specific agent
        context["current_conversation"] = self._extract_conversation(
            current_agent, other_agent
        )
        
        # Here you would call the AI service to get a response
        # return ai_service.get_response(context)
        
        # Placeholder
        return f"This is a placeholder message from agent {current_agent.id}"
    
    def _extract_conversation(self, agent1: Agent, agent2: Agent) -> List[Dict[str, Any]]:
        """Extract the conversation between two specific agents."""
        # Implementation to filter the conversation history for just these two agents
        # This would depend on how conversation history is stored
        return []
```

#### Memory Service (`services/memory_service.py`)
```python
from typing import Dict, Any, List
import logging

from ..models.agent import Agent

logger = logging.getLogger(__name__)

class MemoryService:
    """Handles memory management for agents."""
    
    def build_context(
        self, agent: Agent, game_rules: Dict[str, Any], 
        current_conversation: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Build the context to be provided to an agent."""
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
        """Update an agent's memory with a new message."""
        agent.add_to_memory(message)
```

### 3. Configuration

#### Settings (`config/settings.py`)
```python
import os
import logging
from typing import Dict, Any
from enum import Enum
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Game configuration
DEFAULT_MAX_ROUNDS = int(os.getenv("DEFAULT_MAX_ROUNDS", "10"))
DEFAULT_MESSAGES_PER_ROUND = int(os.getenv("DEFAULT_MESSAGES_PER_ROUND", "3"))

# AI service configuration
AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "")
AI_SERVICE_API_KEY = os.getenv("AI_SERVICE_API_KEY", "")

# Game modes
class GameModeConfig(Enum):
    STANDARD = "standard"
    RETAINED = "retained"
    DIVERSITY = "diversity"
    TARGETED = "targeted"

DEFAULT_GAME_MODE = GameModeConfig[os.getenv("DEFAULT_GAME_MODE", "STANDARD")]

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
```

### 4. CLI Interface

#### Command-Line Interface (`cli.py`)
```python
import click
import logging
import uuid
from typing import List

from .models.game import Game, GameMode
from .models.agent import Agent
from .services.game_service import GameService

logger = logging.getLogger(__name__)

@click.group()
def cli():
    """AI Secret Trading Game CLI."""
    pass

@cli.command()
@click.option("--mode", type=click.Choice([m.value for m in GameMode]), default=GameMode.STANDARD.value)
@click.option("--agents", "-a", multiple=True, help="Agent names")
@click.option("--secrets", "-s", multiple=True, help="Secrets for each agent")
@click.option("--rounds", "-r", default=10, help="Maximum number of rounds")
@click.option("--messages", "-m", default=3, help="Messages per round")
@click.option("--targeted-secret", help="Targeted secret (for targeted mode)")
def run_game(mode, agents, secrets, rounds, messages, targeted_secret):
    """Run a game with the specified parameters."""
    if len(agents) != len(secrets):
        click.echo("Error: Number of agents must match number of secrets")
        return
    
    # Create agents
    game_agents = []
    for i, (name, secret) in enumerate(zip(agents, secrets)):
        agent = Agent(
            id=str(uuid.uuid4()),
            name=name,
            secret=secret
        )
        game_agents.append(agent)
    
    # Create and run game
    game_service = GameService()
    game = game_service.create_game(
        agents=game_agents,
        mode=GameMode(mode),
        max_rounds=rounds,
        messages_per_round=messages,
        targeted_secret=targeted_secret
    )
    
    click.echo(f"Starting game in {mode} mode with {len(agents)} agents")
    
    # Run all rounds
    for _ in range(rounds):
        round_result = game_service.run_round(game)
        click.echo(f"Round {round_result['round']} completed")
    
    # Calculate and display final scores
    final_scores = game_service.scoring_service.calculate_final_scores(game)
    
    click.echo("\nGame Over - Final Scores:")
    for agent in game.agents:
        click.echo(f"{agent.name}: {final_scores[agent.id]}")

if __name__ == "__main__":
    cli()
```

## Implementation Considerations

### AI Model Integration

The implementation provides a flexible framework for integrating different AI models:

1. **Model Agnostic**: The design allows integration with any AI model or API by extending the `AgentService` class
2. **Context Construction**: The memory system provides all necessary context (chat history, secrets, rules) to the AI model
3. **Response Validation**: Add middleware for checking if secrets are revealed in responses
4. **Prompt Engineering**: Consider crafting specific system prompts for each game mode to guide agent behavior
5. **Error Handling**: Add fallback mechanisms for API failures or malformed responses

### Memory Management

The memory system is kept simple and unopinionated as requested:

1. **Context Structure**: Each agent receives `{chat_history, secret(s), rules for game}`
2. **Memory Efficiency**: For long games, consider implementing memory pruning strategies
3. **Serialization**: JSON-serializable format makes it easy to store and transmit
4. **Flexibility**: The architecture allows for future extensions like episodic memory or semantic memory

### Extensibility

The modular design makes it easy to extend the system:

1. **New Game Modes**: Add new modes by extending the `GameMode` enum and scoring methods
2. **Custom Rulesets**: The rules dictionary can be extended with additional parameters
3. **Agent Personalities**: Implement different agent profiles with varying strategies
4. **Tournament Support**: Add tournament orchestration on top of the game service
5. **Analytics**: Track strategies and outcomes across multiple games

### Deployment Options

The system can be deployed in multiple ways:

1. **CLI Application**: Run locally for testing and development
2. **Web Service**: Add FastAPI endpoints to expose game functionality as a service
3. **Async Processing**: Use asynchronous processing for handling multiple games simultaneously
4. **Database Integration**: Add persistence layer for long-running games

## Development Roadmap

1. **MVP Implementation**
   - Core game mechanics
   - Basic CLI interface
   - Integration with one AI provider

2. **Testing & Validation**
   - Unit tests for core logic
   - Integration tests with mock AI
   - Performance testing for larger games

3. **Enhanced Features**
   - Web interface for visualization 
   - Tournament mode
   - Extended analytics
   - User authentication for human players

4. **Deployment & Scaling**
   - Containerization
   - Horizontal scaling
   - Monitoring
   - Cost optimization

## Setup Instructions

1. **Initial Setup**
   ```bash
   # Install Rye
   curl -sSf https://rye-up.com/get | bash
   
   # Clone repository and navigate to project directory
   git clone <repository-url>
   cd ai_secret_game
   
   # Initialize the project with Rye
   rye init
   
   # Add dependencies
   rye add click python-dotenv
   ```

2. **Environment Setup**
   ```bash
   # Create .env file
   echo "LOG_LEVEL=INFO" > .env
   echo "DEFAULT_GAME_MODE=STANDARD" >> .env
   echo "AI_SERVICE_API_KEY=your-api-key" >> .env
   ```

3. **Running the Game**
   ```bash
   # Run a game
   python -m ai_secret_game.cli run-game --agents "Agent1" "Agent2" --secrets "Secret1" "Secret2"
   ```

## Conclusion

This implementation plan provides a robust, modular, and extensible system for implementing the AI secret trading game. The design follows Python best practices, emphasizes clean separation of concerns, and provides flexibility for different game modes and AI integrations.

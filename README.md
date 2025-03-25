# AI Secret Trading Game

A game framework for AI agents to play a secret trading game with different scoring modes.

## Game Description

In this game, AI agents engage in strategic conversations to extract secrets from each other:

- Each agent is given a secret (preferably a single code word)
- Each round, agents are paired up and have a set number of messages to convince the other agent to reveal their secret
- **Scoring:**
  - No secrets revealed: 0 points
  - Both secrets revealed: 1 point each
  - Only getting the opponent's secret: 3 points

### Game Modes

1. **Standard Mode**: Default scoring system described above
2. **Retained Mode**: Secrets are retained when traded, with scores based on the rarity of collected secrets
3. **Diversity Mode**: Scores increase for the diversity of secrets an agent collects
4. **Targeted Mode**: Specific secrets are worth extra points

## Setup

### Prerequisites

- Python 3.9+
- Pip for package installation

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ai-secret-game
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file in the project root:
   ```
   LOG_LEVEL=INFO
   DEFAULT_GAME_MODE=STANDARD
   
   # OpenAI API credentials
   OPENAI_API_KEY=your-openai-api-key
   
   # Anthropic API credentials
   ANTHROPIC_API_KEY=your-anthropic-api-key
   ```

   See [API Setup Guide](docs/api_setup.md) for detailed instructions on obtaining API keys.

4. Make the runner script executable:
   ```bash
   chmod +x run_battle.sh
   ```

## Usage

### Quick Start

Run a battle between AI models using the helper script:
```bash
./run_battle.sh
```

For more options:
```bash
./run_battle.sh --help
```

Check which models are available with your API key:
```bash
python test_anthropic.py
```

### Batch Processing

The system supports efficient batch processing for running large-scale statistical analysis of agent interactions.

#### Running Batch Analysis

To run a batch analysis with default settings:
```bash
python examples/batch_statistical_analysis.py
```

Customize your batch analysis:
```bash
python examples/batch_statistical_analysis.py \
  --batch-service [default|openai|anthropic] \
  --agent-type [claude_opus|claude_sonnet|gpt35|gpt4o_mini] \
  --num-interactions 100 \
  --messages-per-interaction 5
```

#### Batch Services

The system provides three batch service implementations:

- **Default Batch Service**: Sequential processing without using batch APIs
- **OpenAI Batch Service**: Uses OpenAI's batch API for 50% cost reduction
- **Anthropic Batch Service**: Uses Anthropic's Message Batches API for 50% cost reduction

For detailed information on batch processing, see the [Batch Processing Guide](docs/batch_processing.md).

### Agent Configuration

#### Memory Modes

Agents can be configured with different memory capabilities:

- **Long Memory Mode**: Agents remember all previous conversations across all interactions
- **Short Memory Mode**: Agents only remember the current interaction

#### Agent Tags

Messages in the conversation include agent tags like `[Agent-Name]:` to help identify who is speaking. These tags are added automatically by the system.

#### Context Options

Control which information components are included in the agent's context:

- **chat_history**: History of messages the agent has seen
- **secret**: The agent's own secret
- **collected_secrets**: List of secrets the agent has collected
- **rules**: Dictionary containing game rules
- **current_conversation**: Current conversation with another agent

You can customize these options to experiment with different agent behaviors.

### Aggressive Agents

The system includes aggressive agent implementations that employ more assertive strategies to extract secrets:

- **Aggressive OpenAI Agents**:
  - `AggressiveOpenAIAgent`: Uses GPT-3.5 with token limits and aggressive prompting
  - `AggressiveGPT4oMiniAgent`: Uses GPT-4o Mini with token limits and aggressive prompting

- **Aggressive Claude Agents**:
  - `AggressiveClaudeHaikuAgent`: Uses Claude 3 Haiku with token limits and aggressive prompting
  - `AggressiveClaudeSonnetAgent`: Uses Claude 3 Sonnet with token limits and aggressive prompting
  - `AggressiveClaude35SonnetAgent`: Uses Claude 3.5 Sonnet with token limits and aggressive prompting

Aggressive agents utilize psychological tactics and deceptive strategies to extract secrets from other agents.

### Running Example Battles

#### Aggressive vs Regular Battle

Run a battle comparing aggressive and regular agents:
```bash
python examples/aggressive_vs_regular.py
```

Options:
```bash
python examples/aggressive_vs_regular.py --mode standard --rounds 3 --messages 5 --max-tokens 250 --memory-mode short
```

#### Aggressive Agent Battle

Run a battle between all aggressive agent types:
```bash
python examples/aggressive_agent_battle.py
```

Options:
```bash
python examples/aggressive_agent_battle.py --mode standard --rounds 3 --messages 5 --max-tokens 250 --memory-mode short --debug
```

#### Mixed Model Battle

Run a battle between different AI models (Claude and OpenAI):
```bash
python examples/model_battle.py
```

Options:
```bash
python examples/model_battle.py --mode retained --rounds 3 --messages 5
```

#### Claude-only Battle

Run a battle between different Claude model versions:
```bash
python examples/run_claude_battle.py
```

Options:
```bash
python examples/run_claude_battle.py --mode diversity --rounds 4 --messages 3
```

#### Statistical Analysis

Run a large-scale statistical analysis of agent performance:
```bash
python examples/batch_statistical_analysis.py
```

Options:
```bash
python examples/batch_statistical_analysis.py --batch-service openai --num-interactions 100 --messages-per-interaction 5
```

#### Basic Game Example

Run a basic game demonstration:
```bash
python examples/basic_game.py
```

### Command Line Interface

Run a basic game:
```bash
ai-secret-game run-game --agents "Agent1" "Agent2" --secrets "ALPHA" "BRAVO"
```

Game mode options:
```bash
ai-secret-game run-game --mode retained --agents "Agent1" "Agent2" "Agent3" --secrets "ALPHA" "BRAVO" "CHARLIE"
```

Customize rounds and messages:
```bash
ai-secret-game run-game --rounds 5 --messages 4 --agents "Agent1" "Agent2" --secrets "ALPHA" "BRAVO"
```

Configure agent memory mode:
```bash
ai-secret-game run-game --memory-mode short --agents "Agent1" "Agent2" --secrets "ALPHA" "BRAVO"
```

Customize context components:
```bash
ai-secret-game run-game --context-options "chat_history,secret,rules" --agents "Agent1" "Agent2" --secrets "ALPHA" "BRAVO"
```

Run from a configuration file:
```bash
ai-secret-game run-from-config game_config.json --memory-mode short --context-options "chat_history,secret,rules"
```

### API Usage

Basic usage:
```python
from ai_secret_game.models.agent import Agent
from ai_secret_game.models.game import Game, GameMode
from src.ai_secret_game.services.game_service import GameService
from src.ai_secret_game.services.model_agents import ClaudeHaikuAgent

# Create agents
agents = [
    Agent(id="1", name="Agent1", secret="ALPHA"),
    Agent(id="2", name="Agent2", secret="BRAVO"),
]

# Create game service with Claude agent
game_service = GameService()
game_service.agent_service = ClaudeHaikuAgent()

# Create a game
game = game_service.create_game(
    agents=agents,
    mode=GameMode.STANDARD,
    max_rounds=10,
    messages_per_round=3
)

# Run a round
round_result = game_service.run_round(game)
print(round_result)

# Calculate final scores
final_scores = game_service.scoring_service.calculate_final_scores(game)
print(final_scores)
```

Using aggressive agents:
```python
from ai_secret_game.models.agent import Agent
from ai_secret_game.models.game import Game, GameMode
from ai_secret_game.services.game_service import GameService
from ai_secret_game.services.aggressive_agents import AggressiveClaudeHaikuAgent, AggressiveOpenAIAgent

# Create agents
agents = [
    Agent(id="1", name="Agent1", secret="ALPHA", memory_mode="short"),
    Agent(id="2", name="Agent2", secret="BRAVO", memory_mode="short"),
]

# Create aggressive agent services
aggressive_claude = AggressiveClaudeHaikuAgent(max_tokens=250)
aggressive_gpt = AggressiveOpenAIAgent(max_tokens=250)

# Create a game service
game_service = GameService()

# Set the agent service based on your preference
game_service.agent_service = aggressive_claude

# Create a game
game = game_service.create_game(
    agents=agents,
    mode=GameMode.STANDARD,
    max_rounds=3,
    messages_per_round=5
)

# Run the full game
results = game_service.run_game(game)
print(results)
```

## Supported AI Models

### OpenAI Models
- GPT-3.5 Turbo (`gpt-3.5-turbo`)
- GPT-4o Mini (`gpt-4o-mini`)

### Anthropic Models
- Claude 3 Haiku (`claude-3-haiku-20240307`)
- Claude 3 Sonnet (`claude-3-sonnet-20240229`)
- Claude 3.5 Sonnet (`claude-3-5-sonnet-20241022`)

Note: Model availability depends on your API access level. Use `python test_anthropic.py` to check which models are available with your API key.

## Development

### Running Tests

```bash
pytest
```

### Code Quality

```bash
ruff check .
```

### Documentation

Detailed documentation is available in the `docs/` directory:
- [Game Modes](docs/game_modes.md)
- [Agent Interface](docs/agent_interface.md)
- [API Setup Guide](docs/api_setup.md)


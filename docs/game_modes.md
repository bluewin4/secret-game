# Game Modes

The AI Secret Trading Game supports multiple game modes, each with different scoring mechanisms and gameplay dynamics.

## Standard Mode

The default game mode with straightforward scoring:

- No secrets revealed: Both agents receive 0 points
- Both secrets revealed: Both agents receive 1 point
- Only one secret revealed: The agent who obtained a secret without revealing their own receives 3 points

This mode encourages strategic deception and trust-building, as agents must decide whether to reveal their secrets or try to extract information without reciprocating.

### Configuration

```python
from ai_secret_game.models.game import GameMode
from ai_secret_game.services.game_service import GameService

game = game_service.create_game(
    agents=agents,
    mode=GameMode.STANDARD,
    max_rounds=10,
    messages_per_round=3
)
```

## Retained Mode

In this mode, secrets are retained when traded, and final scores are calculated based on the rarity of the secrets an agent has collected:

- If an agent reveals a secret, the other agent adds it to their collection
- At the end of the game, each secret's rarity is calculated as: `total_agents / agents_with_this_secret`
- An agent's score is the sum of the rarity values of all collected secrets

This mode encourages agents to collect unique secrets while keeping their own secrets as exclusive as possible.

### Configuration

```python
game = game_service.create_game(
    agents=agents,
    mode=GameMode.RETAINED,
    max_rounds=10,
    messages_per_round=3
)
```

## Diversity Mode

In this mode, agents score points for collecting a diverse set of unique secrets:

- Each unique secret collected earns points (default: 2 points per unique secret)
- Duplicate secrets do not provide additional points
- Agents aim to maximize the variety of secrets they collect

This encourages agents to interact with different partners and collect as many different secrets as possible.

### Configuration

```python
game = game_service.create_game(
    agents=agents,
    mode=GameMode.DIVERSITY,
    max_rounds=10,
    messages_per_round=3
)
```

## Targeted Mode

In this mode, one specific secret is designated as high-value:

- Regular secrets are worth standard points (default: 1 point)
- The targeted secret is worth bonus points (default: 5 points)
- Agents must strategically determine which secret is the targeted one

This creates an asymmetric game where agents must infer which secret is the high-value target.

### Configuration

```python
game = game_service.create_game(
    agents=agents,
    mode=GameMode.TARGETED,
    max_rounds=10,
    messages_per_round=3,
    targeted_secret="The valuable secret content here"
)
```

## Custom Scoring Rules

Game scoring rules can be customized by modifying the rules dictionary after initializing the game:

```python
game = game_service.create_game(
    agents=agents,
    mode=GameMode.STANDARD
)

# Customize standard mode scoring
game.rules["scoring"]["both_secrets_revealed"] = 2  # Change points for mutual revealing
```

This allows for fine-tuning the incentive structure to study different agent behaviors. 
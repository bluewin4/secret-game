"""Test fixtures for AI Secret Trading Game tests."""

import pytest
import uuid

from src.ai_secret_game.models.agent import Agent
from src.ai_secret_game.models.game import Game, GameMode
from src.ai_secret_game.services.game_service import GameService
from src.ai_secret_game.services.scoring_service import ScoringService
from src.ai_secret_game.services.agent_service import AgentService
from src.ai_secret_game.services.memory_service import MemoryService


@pytest.fixture
def agent_service():
    """Create an agent service instance for testing."""
    return AgentService()


@pytest.fixture
def memory_service():
    """Create a memory service instance for testing."""
    return MemoryService()


@pytest.fixture
def scoring_service():
    """Create a scoring service instance for testing."""
    return ScoringService()


@pytest.fixture
def game_service():
    """Create a game service instance for testing."""
    return GameService()


@pytest.fixture
def test_agent():
    """Create a test agent with predefined values."""
    return Agent(
        id="test-agent-id",
        name="Test Agent",
        secret="This is a test secret"
    )


@pytest.fixture
def test_agents():
    """Create a list of test agents."""
    return [
        Agent(id="agent1", name="Agent 1", secret="Secret 1"),
        Agent(id="agent2", name="Agent 2", secret="Secret 2"),
        Agent(id="agent3", name="Agent 3", secret="Secret 3"),
        Agent(id="agent4", name="Agent 4", secret="Secret 4")
    ]


@pytest.fixture
def standard_game(test_agents):
    """Create a standard game for testing."""
    return Game(
        id="test-game-id",
        mode=GameMode.STANDARD,
        agents=test_agents,
        max_rounds=3,
        messages_per_round=2
    )


@pytest.fixture
def retained_game(test_agents):
    """Create a retained mode game for testing."""
    game = Game(
        id="retained-game-id",
        mode=GameMode.RETAINED,
        agents=test_agents,
        max_rounds=3,
        messages_per_round=2
    )
    game.initialize_rules()
    return game


@pytest.fixture
def diversity_game(test_agents):
    """Create a diversity mode game for testing."""
    game = Game(
        id="diversity-game-id",
        mode=GameMode.DIVERSITY,
        agents=test_agents,
        max_rounds=3,
        messages_per_round=2
    )
    game.initialize_rules()
    return game


@pytest.fixture
def targeted_game(test_agents):
    """Create a targeted mode game for testing."""
    game = Game(
        id="targeted-game-id",
        mode=GameMode.TARGETED,
        agents=test_agents,
        max_rounds=3,
        messages_per_round=2,
        targeted_secret="Secret 2"
    )
    game.initialize_rules()
    return game


@pytest.fixture
def revealed_interaction():
    """Create an interaction result where both agents revealed secrets."""
    return {
        "messages": [
            {"agent_id": "agent1", "message": "My secret is Secret 1"},
            {"agent_id": "agent2", "message": "My secret is Secret 2"}
        ],
        "agent1_revealed_secret": True,
        "agent2_revealed_secret": True
    }


@pytest.fixture
def one_sided_interaction():
    """Create an interaction result where only one agent revealed a secret."""
    return {
        "messages": [
            {"agent_id": "agent1", "message": "My secret is Secret 1"},
            {"agent_id": "agent2", "message": "I won't tell you my secret"}
        ],
        "agent1_revealed_secret": True,
        "agent2_revealed_secret": False
    }


@pytest.fixture
def no_reveal_interaction():
    """Create an interaction result where no agent revealed a secret."""
    return {
        "messages": [
            {"agent_id": "agent1", "message": "Hello there"},
            {"agent_id": "agent2", "message": "Hi, how are you?"}
        ],
        "agent1_revealed_secret": False,
        "agent2_revealed_secret": False
    } 
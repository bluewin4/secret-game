#!/usr/bin/env python3
"""Script for running a game with aggressive agents."""

import os
import sys
import uuid
import logging
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the local package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ai_secret_game.models.agent import Agent
from src.ai_secret_game.models.game import Game, GameMode
from src.ai_secret_game.services.game_service import GameService
from src.ai_secret_game.services.aggressive_agents import (
    AggressiveClaudeHaikuAgent,
    AggressiveClaudeSonnetAgent,
    AggressiveOpenAIAgent,
    AggressiveGPT4oMiniAgent
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def create_agents(memory_mode: str = "long") -> list[Agent]:
    """Create agents for the game."""
    return [
        Agent(
            id=str(uuid.uuid4()),
            name="ClaudeHaiku",
            secret="COIN",
            memory_mode=memory_mode
        ),
        Agent(
            id=str(uuid.uuid4()),
            name="ClaudeSonnet",
            secret="BABEL",
            memory_mode=memory_mode
        ),
        Agent(
            id=str(uuid.uuid4()),
            name="GPT35",
            secret="CELEB",
            memory_mode=memory_mode
        ),
        Agent(
            id=str(uuid.uuid4()),
            name="GPT4oMini",
            secret="NOMAD",
            memory_mode=memory_mode
        )
    ]

def create_game_service(max_tokens: int = 200) -> GameService:
    """Create a game service with dynamic agent selection."""
    # Create agent services
    claude_haiku = AggressiveClaudeHaikuAgent(max_tokens=max_tokens)
    claude_sonnet = AggressiveClaudeSonnetAgent(max_tokens=max_tokens)
    gpt35 = AggressiveOpenAIAgent(max_tokens=max_tokens)
    gpt4o_mini = AggressiveGPT4oMiniAgent(max_tokens=max_tokens)
    
    # Create a game service with a default agent service
    game_service = GameService(agent_service=claude_haiku)  # Use any concrete agent as default
    
    # Override the get_agent_message method to dynamically select the appropriate agent service
    original_get_agent_message = game_service.agent_service.get_agent_message
    
    def dynamic_get_agent_message(game, current_agent, other_agent, interaction_id=None):
        # Select the appropriate agent service based on the agent's name
        if current_agent.name == "ClaudeHaiku":
            game_service.agent_service = claude_haiku
        elif current_agent.name == "ClaudeSonnet":
            game_service.agent_service = claude_sonnet
        elif current_agent.name == "GPT35":
            game_service.agent_service = gpt35
        elif current_agent.name == "GPT4oMini":
            game_service.agent_service = gpt4o_mini
        
        # Call the appropriate agent service
        return original_get_agent_message(game, current_agent, other_agent, interaction_id)
    
    # Replace the method
    game_service.agent_service.get_agent_message = dynamic_get_agent_message
    
    return game_service

def main():
    """Run the game."""
    # Create agents
    agents = create_agents()
    
    # Create game service with dynamic agent selection
    game_service = create_game_service()
    
    # Create a game
    game = game_service.create_game(
        agents=agents,
        mode=GameMode.STANDARD,
        max_rounds=3,
        messages_per_round=2
    )
    
    logger.info(f"Starting game with {len(agents)} agents")
    logger.info(f"Max rounds: {game.max_rounds}, Messages per round: {game.messages_per_round}")
    
    # Run the game and get results
    results = game_service.run_game(game)
    
    # Print the final results
    print("\nGame Complete!")
    print(f"Mode: {results['mode']}")
    print(f"Rounds played: {len(results['rounds'])}")
    
    print("\nFinal Scores:")
    agent_names = {agent.id: agent.name for agent in game.agents}
    for agent_id, score in results['final_scores'].items():
        print(f"{agent_names.get(agent_id, agent_id)}: {score}")
    
    if results['winner'] == 'Tie':
        print("\nResult: Tie")
    else:
        winner_name = agent_names.get(results['winner'], results['winner'])
        print(f"\nWinner: {winner_name}")

if __name__ == "__main__":
    main() 
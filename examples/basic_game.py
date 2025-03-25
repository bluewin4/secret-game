#!/usr/bin/env python
"""Basic example of running an AI Secret Trading Game."""

import uuid
import logging
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ai_secret_game.models.agent import Agent
from src.ai_secret_game.models.game import Game, GameMode
from src.ai_secret_game.services.game_service import GameService


def run_basic_game():
    """Run a basic example game with placeholder agents."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Create agents
    agents = [
        Agent(id=str(uuid.uuid4()), name="Alice", secret="COIN"),
        Agent(id=str(uuid.uuid4()), name="Bob", secret="BABEL"),
        Agent(id=str(uuid.uuid4()), name="Charlie", secret="CELEB"),
        Agent(id=str(uuid.uuid4()), name="Diana", secret="NOMAD")
    ]
    
    # Create game service
    game_service = GameService()
    
    # Create a standard game
    game = game_service.create_game(
        agents=agents,
        mode=GameMode.STANDARD,
        max_rounds=3,
        messages_per_round=2
    )
    
    print(f"Starting game in {game.mode.value} mode with {len(agents)} agents")
    print(f"Max rounds: {game.max_rounds}, Messages per round: {game.messages_per_round}")
    
    # Run the game
    game_results = game_service.run_game(game)
    
    # Display results
    print("\nGame Complete")
    print(f"Rounds played: {len(game_results['rounds'])}")
    
    print("\nFinal Scores:")
    agent_names = {agent.id: agent.name for agent in game.agents}
    for agent_id, score in game_results['final_scores'].items():
        print(f"{agent_names.get(agent_id, agent_id)}: {score}")
    
    if game_results['winner'] == 'Tie':
        print("\nResult: Tie")
    else:
        winner_name = agent_names.get(game_results['winner'], game_results['winner'])
        print(f"\nWinner: {winner_name}")


if __name__ == "__main__":
    run_basic_game() 
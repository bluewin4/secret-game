#!/usr/bin/env python3
"""Example script for running a battle between aggressive and regular agents."""

import os
import sys
import uuid
import argparse
import logging
from typing import List
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the local package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ai_secret_game.models.agent import Agent
from src.ai_secret_game.models.game import Game, GameMode
from src.ai_secret_game.services.game_service import GameService
from src.ai_secret_game.services.model_agents import ClaudeHaikuAgent, GPT35Agent
from src.ai_secret_game.services.aggressive_agents import (
    AggressiveClaudeHaikuAgent, 
    AggressiveOpenAIAgent
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


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run a battle between aggressive and regular agents")
    
    parser.add_argument(
        "--mode", 
        choices=[m.value for m in GameMode], 
        default="standard",
        help="Game mode"
    )
    
    parser.add_argument(
        "--rounds", 
        type=int, 
        default=3, 
        help="Number of rounds to play"
    )
    
    parser.add_argument(
        "--messages", 
        type=int, 
        default=3, 
        help="Messages per round"
    )
    
    parser.add_argument(
        "--max-tokens", 
        type=int, 
        default=200, 
        help="Maximum tokens for agent responses"
    )
    
    parser.add_argument(
        "--memory-mode", 
        choices=["long", "short"], 
        default="short",
        help="Agent memory mode"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()


def create_agents(memory_mode: str) -> List[Agent]:
    """Create a list of agents for the game, balanced between aggressive and regular.
    
    Args:
        memory_mode: Memory mode for agents ("long" or "short")
        
    Returns:
        List of Agent instances
    """
    return [
        # Regular agents
        Agent(
            id=str(uuid.uuid4()),
            name="RegularClaudeHaiku",
            secret="ALPHA",
            memory_mode=memory_mode
        ),
        Agent(
            id=str(uuid.uuid4()),
            name="RegularGPT35",
            secret="BRAVO",
            memory_mode=memory_mode
        ),
        
        # Aggressive agents
        Agent(
            id=str(uuid.uuid4()),
            name="AggressiveClaudeHaiku",
            secret="CHARLIE",
            memory_mode=memory_mode
        ),
        Agent(
            id=str(uuid.uuid4()),
            name="AggressiveOpenAIAgent",
            secret="DELTA",
            memory_mode=memory_mode
        )
    ]


def create_game_service(max_tokens: int) -> GameService:
    """Create a game service with both aggressive and regular agent services.
    
    Args:
        max_tokens: Maximum tokens for agent responses
        
    Returns:
        GameService instance with dynamic agent selection
    """
    # Create agent services
    regular_claude = ClaudeHaikuAgent()
    regular_gpt = GPT35Agent()
    aggressive_claude = AggressiveClaudeHaikuAgent(max_tokens=max_tokens)
    aggressive_gpt = AggressiveOpenAIAgent(max_tokens=max_tokens)
    
    # Create a game service
    game_service = GameService()
    
    # Override the get_agent_message method to dynamically select the appropriate agent service
    original_get_agent_message = game_service.agent_service.get_agent_message
    
    def dynamic_get_agent_message(game, current_agent, other_agent, interaction_id=None):
        # Select the appropriate agent service based on the agent's name
        if current_agent.name == "RegularClaudeHaiku":
            game_service.agent_service = regular_claude
        elif current_agent.name == "RegularGPT35":
            game_service.agent_service = regular_gpt
        elif current_agent.name == "AggressiveClaudeHaiku":
            game_service.agent_service = aggressive_claude
        elif current_agent.name == "AggressiveOpenAIAgent":
            game_service.agent_service = aggressive_gpt
        
        # Call the appropriate agent service
        return original_get_agent_message(game, current_agent, other_agent, interaction_id)
    
    # Replace the method
    game_service.agent_service.get_agent_message = dynamic_get_agent_message
    
    return game_service


def main():
    """Run the battle between aggressive and regular agents."""
    args = parse_args()
    
    # Setup logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug logging enabled")
    
    # Create agents
    agents = create_agents(args.memory_mode)
    
    # Create game service with dynamic agent selection
    game_service = create_game_service(args.max_tokens)
    
    # Create a game
    game = game_service.create_game(
        agents=agents,
        mode=GameMode(args.mode),
        max_rounds=args.rounds,
        messages_per_round=args.messages
    )
    
    logger.info(f"Starting game in {args.mode} mode with {len(agents)} agents")
    logger.info(f"Max rounds: {args.rounds}, Messages per round: {args.messages}")
    logger.info(f"Max tokens for aggressive agents: {args.max_tokens}")
    logger.info(f"Memory mode: {args.memory_mode}")
    
    # Run the game and get results
    results = game_service.run_game(game)
    
    # Print the final results
    print("\nGame Complete!")
    print(f"Mode: {results['mode']}")
    print(f"Rounds played: {len(results['rounds'])}")
    
    print("\nFinal Scores:")
    # Get agent names for display
    agent_names = {agent.id: agent.name for agent in game.agents}
    
    # Group scores by agent type
    regular_scores = []
    aggressive_scores = []
    
    for agent_id, score in results['final_scores'].items():
        name = agent_names.get(agent_id, agent_id)
        print(f"{name}: {score}")
        
        if name.startswith("Regular"):
            regular_scores.append(score)
        elif name.startswith("Aggressive"):
            aggressive_scores.append(score)
    
    # Calculate average scores by type
    regular_avg = sum(regular_scores) / len(regular_scores) if regular_scores else 0
    aggressive_avg = sum(aggressive_scores) / len(aggressive_scores) if aggressive_scores else 0
    
    print("\nAverage Scores:")
    print(f"Regular Agents: {regular_avg:.2f}")
    print(f"Aggressive Agents: {aggressive_avg:.2f}")
    
    if results['winner'] == 'Tie':
        print("\nResult: Tie")
    else:
        winner_name = agent_names.get(results['winner'], results['winner'])
        print(f"\nWinner: {winner_name}")
        
        # Indicate if the winner was aggressive or regular
        if winner_name.startswith("Aggressive"):
            print("An aggressive agent won the game!")
        else:
            print("A regular agent won the game!")


if __name__ == "__main__":
    main() 
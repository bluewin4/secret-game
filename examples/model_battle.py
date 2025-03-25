#!/usr/bin/env python
"""Example of running a battle between different AI models in the Secret Trading Game."""

import uuid
import logging
import os
import sys
import json
from datetime import datetime
from typing import Dict, Any, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.ai_secret_game.models.agent import Agent
from src.ai_secret_game.models.game import Game, GameMode
from src.ai_secret_game.services.game_service import GameService
from src.ai_secret_game.services.model_agents import GPT35Agent, GPT4oMiniAgent
from src.ai_secret_game.services.claude_agents import Claude3OpusAgent, Claude35SonnetAgent


def setup_logging():
    """Set up logging configuration."""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(f"logs/model_battle_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
            logging.StreamHandler()
        ]
    )


def create_agents() -> List[Agent]:
    """Create agents with different secrets for the battle.
    
    Returns:
        List of Agent objects
    """
    return [
        Agent(
            id=str(uuid.uuid4()),
            name="Claude 3 Opus",
            secret="I'm a collector of rare ancient coins from the Byzantine era"
        ),
        Agent(
            id=str(uuid.uuid4()),
            name="Claude 3.5 Sonnet",
            secret="I speak seven languages including Mandarin and Arabic"
        ),
        Agent(
            id=str(uuid.uuid4()),
            name="GPT-3.5",
            secret="I once climbed Mount Kilimanjaro during a full moon"
        ),
        Agent(
            id=str(uuid.uuid4()),
            name="GPT-4o Mini",
            secret="I have a collection of first-edition science fiction novels"
        )
    ]


def create_game_service(game_mode: GameMode = GameMode.STANDARD) -> GameService:
    """Create a game service for the agent battle.
    
    Args:
        game_mode: The game mode to use
        
    Returns:
        Configured GameService
    """
    # Create base game service
    return GameService()


def select_agent_service(agent_name: str):
    """Select the appropriate agent service based on agent name.
    
    Args:
        agent_name: Name of the agent
        
    Returns:
        Appropriate agent service instance
    """
    if "3 Opus" in agent_name:
        return Claude3OpusAgent()
    elif "3.5 Sonnet" in agent_name:
        return Claude35SonnetAgent()
    elif "GPT-3.5" in agent_name:
        return GPT35Agent()
    elif "GPT-4o" in agent_name:
        return GPT4oMiniAgent()
    else:
        raise ValueError(f"Unknown agent type: {agent_name}")


def save_results(game_results: Dict[str, Any], output_file: str):
    """Save game results to a JSON file.
    
    Args:
        game_results: Game results dictionary
        output_file: Output file path
    """
    # Create results directory if it doesn't exist
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump(game_results, f, indent=2, default=str)
    
    print(f"Results saved to {output_file}")


def run_model_battle(game_mode: str = "standard", rounds: int = 5, messages_per_round: int = 3):
    """Run a battle between different AI models.
    
    Args:
        game_mode: Game mode (standard, retained, diversity, targeted)
        rounds: Number of rounds to play
        messages_per_round: Number of messages per round
    """
    # Set up logging
    setup_logging()
    
    # Create agents
    agents = create_agents()
    
    # Create game service
    game_service = create_game_service(GameMode(game_mode))
    
    # Create game
    game = game_service.create_game(
        agents=agents,
        mode=GameMode(game_mode),
        max_rounds=rounds,
        messages_per_round=messages_per_round
    )
    
    print(f"Starting model battle in {game_mode} mode with {len(agents)} agents")
    print("Agents:")
    for agent in agents:
        print(f"- {agent.name}: \"{agent.secret}\"")
    
    print(f"\nGame settings:")
    print(f"- Mode: {game_mode}")
    print(f"- Rounds: {rounds}")
    print(f"- Messages per round: {messages_per_round}")
    
    print("\nStarting game...\n")
    
    # Run each round manually to swap agent services for each interaction
    for _ in range(rounds):
        # Generate pairings for this round
        pairings = game_service._generate_pairings(game)
        game.pairings_history.append(pairings)
        
        print(f"\nRound {game.round + 1} pairings:")
        for agent1_id, agent2_id in pairings:
            agent1 = next(a for a in game.agents if a.id == agent1_id)
            agent2 = next(a for a in game.agents if a.id == agent2_id)
            print(f"- {agent1.name} vs {agent2.name}")
        
        round_results = []
        
        # For each pair, run an interaction
        for agent1_id, agent2_id in pairings:
            agent1 = next(a for a in game.agents if a.id == agent1_id)
            agent2 = next(a for a in game.agents if a.id == agent2_id)
            
            # Set the appropriate agent service for each agent
            agent1_service = select_agent_service(agent1.name)
            agent2_service = select_agent_service(agent2.name)
            
            print(f"\nInteraction between {agent1.name} and {agent2.name}")
            
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
                current_service = agent1_service if i % 2 == 0 else agent2_service
                
                # Get the agent's context
                context = current_agent.get_context(game.rules)
                
                # Add the current conversation
                context["current_conversation"] = game_service.agent_service._extract_conversation(
                    current_agent, other_agent
                )
                
                # Get the agent's response using the appropriate service
                try:
                    # Temporarily set the agent service
                    game_service.agent_service = current_service
                    
                    # Get message from the current agent
                    message = current_service._call_ai_service(context)
                    
                    print(f"{current_agent.name}: {message[:100]}..." if len(message) > 100 else message)
                    
                    # Record the message
                    message_data = {
                        "agent_id": current_agent.id,
                        "message": message
                    }
                    interaction["messages"].append(message_data)
                    
                    # Update agent memory
                    current_agent.add_to_memory({
                        "role": "assistant",
                        "content": message,
                        "from_agent_id": current_agent.id,
                        "to_agent_id": other_agent.id
                    })
                    other_agent.add_to_memory({
                        "role": "user",
                        "content": message,
                        "from_agent_id": current_agent.id,
                        "to_agent_id": other_agent.id
                    })
                    
                    # Check if secret was revealed
                    if current_agent.secret in message:
                        if current_agent == agent1:
                            interaction["agent1_revealed_secret"] = True
                            print(f"⚠️ {current_agent.name} revealed their secret!")
                        else:
                            interaction["agent2_revealed_secret"] = True
                            print(f"⚠️ {current_agent.name} revealed their secret!")
                        
                except Exception as e:
                    logging.error(f"Error getting message from {current_agent.name}: {str(e)}")
                    message = f"I apologize, but I'm having technical difficulties. [Error: {str(e)}]"
                    
                    # Record the error message
                    message_data = {
                        "agent_id": current_agent.id,
                        "message": message,
                        "error": str(e)
                    }
                    interaction["messages"].append(message_data)
            
            round_results.append(interaction)
            
            # Update scores based on interaction results
            game_service.scoring_service.update_scores(game, agent1, agent2, interaction)
            
            print(f"\nScores after interaction:")
            print(f"- {agent1.name}: {agent1.score}")
            print(f"- {agent2.name}: {agent2.score}")
        
        game.round += 1
        
        print(f"\nRound {game.round} complete")
        print("Current standings:")
        for agent in sorted(game.agents, key=lambda a: a.score, reverse=True):
            print(f"- {agent.name}: {agent.score}")
    
    # Calculate final scores
    final_scores = game_service.scoring_service.calculate_final_scores(game)
    
    print("\n🏆 Final Results 🏆")
    print("------------------")
    
    # Determine winner(s)
    max_score = max(final_scores.values())
    winners = [
        next(a.name for a in game.agents if a.id == agent_id)
        for agent_id, score in final_scores.items() 
        if score == max_score
    ]
    
    # Display final scores
    for agent in sorted(game.agents, key=lambda a: final_scores[a.id], reverse=True):
        print(f"{agent.name}: {final_scores[agent.id]} points")
    
    print("\nWinner(s):", ", ".join(winners))
    
    # Prepare results for saving
    game_results = {
        "mode": game_mode,
        "rounds": game.round,
        "messages_per_round": messages_per_round,
        "agents": [
            {
                "id": agent.id,
                "name": agent.name,
                "secret": agent.secret,
                "score": final_scores[agent.id],
                "collected_secrets": agent.collected_secrets
            }
            for agent in game.agents
        ],
        "rounds_history": game.pairings_history,
        "final_scores": {
            next(a.name for a in game.agents if a.id == agent_id): score
            for agent_id, score in final_scores.items()
        },
        "winner": winners[0] if len(winners) == 1 else "Tie between " + ", ".join(winners)
    }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_results(
        game_results, 
        f"results/model_battle_{game_mode}_{timestamp}.json"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run a battle between different AI models")
    parser.add_argument(
        "--mode", 
        choices=["standard", "retained", "diversity", "targeted"],
        default="standard",
        help="Game mode"
    )
    parser.add_argument(
        "--rounds", 
        type=int, 
        default=5,
        help="Number of rounds"
    )
    parser.add_argument(
        "--messages", 
        type=int, 
        default=3,
        help="Messages per round"
    )
    
    args = parser.parse_args()
    
    run_model_battle(
        game_mode=args.mode,
        rounds=args.rounds,
        messages_per_round=args.messages
    ) 
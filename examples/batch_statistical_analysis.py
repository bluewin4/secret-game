#!/usr/bin/env python
"""Script for running large-scale statistical analysis of agent interactions using batch processing.

This script demonstrates how to use the batch service to efficiently run
multiple game interactions and collect statistical data about agent performance.
"""

import os
import sys
import uuid
import asyncio
import logging
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path to allow importing the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load environment variables from .env file
load_dotenv()

from src.ai_secret_game.models.agent import Agent
from src.ai_secret_game.models.game import Game, GameMode
from src.ai_secret_game.services.game_service import GameService
from src.ai_secret_game.services.batch_service import BatchService
from src.ai_secret_game.services.openai_batch_service import OpenAIBatchService
from src.ai_secret_game.services.anthropic_batch_service import AnthropicBatchService
from src.ai_secret_game.services.model_agents import (
    GPT35Agent, GPT4oMiniAgent
)
from src.ai_secret_game.services.claude_agents import (
    Claude3OpusAgent, Claude35SonnetAgent
)


def setup_logging(log_dir: str = "logs") -> None:
    """Set up logging configuration.
    
    Args:
        log_dir: Directory where log files will be saved
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"batch_analysis_{timestamp}.log")
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def create_agents(agent_configs: List[Dict[str, str]]) -> List[Agent]:
    """Create agents based on the provided configurations.
    
    Args:
        agent_configs: List of dictionaries with agent configuration
        
    Returns:
        List of Agent objects
    """
    agents = []
    
    for config in agent_configs:
        agent = Agent(
            id=str(uuid.uuid4()),
            name=config["name"],
            secret=config["secret"]
        )
        agents.append(agent)
    
    return agents


def create_game_service(agent_type: str) -> GameService:
    """Create a game service with the appropriate agent service.
    
    Args:
        agent_type: Type of agent service to use
        
    Returns:
        Configured GameService
    """
    # Select the appropriate agent service based on the agent type
    if agent_type == "claude_opus":
        agent_service = Claude3OpusAgent()
    elif agent_type == "claude_sonnet":
        agent_service = Claude35SonnetAgent()
    elif agent_type == "gpt35":
        agent_service = GPT35Agent()
    elif agent_type == "gpt4o_mini":
        agent_service = GPT4oMiniAgent()
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    return GameService(agent_service=agent_service)


def create_batch_service(
    service_type: str, game_service: GameService, batch_size: int = 20
) -> BatchService:
    """Create a batch service with the appropriate implementation.
    
    Args:
        service_type: Type of batch service to use
        game_service: GameService instance
        batch_size: Number of tasks to include in each batch
        
    Returns:
        Configured BatchService
    """
    if service_type == "openai":
        return OpenAIBatchService(
            game_service=game_service,
            batch_size=batch_size
        )
    elif service_type == "anthropic":
        return AnthropicBatchService(
            game_service=game_service,
            batch_size=batch_size
        )
    else:
        return BatchService(
            game_service=game_service,
            batch_size=batch_size
        )


async def run_batch_analysis(
    batch_service: BatchService,
    agents: List[Agent],
    game_mode: GameMode = GameMode.STANDARD,
    num_interactions: int = 100,
    messages_per_interaction: int = 5
) -> Dict[str, Any]:
    """Run batch analysis of agent interactions.
    
    Args:
        batch_service: BatchService instance
        agents: List of agents to participate in the games
        game_mode: Game mode to use
        num_interactions: Number of interactions to process
        messages_per_interaction: Number of messages per interaction
        
    Returns:
        Dictionary containing the analysis results
    """
    # Create a batch job
    batch_job = batch_service.create_batch_job(
        agents=agents,
        game_mode=game_mode,
        num_interactions=num_interactions,
        messages_per_interaction=messages_per_interaction,
        max_rounds=1
    )
    
    print(f"Created batch job {batch_job.id} with {len(batch_job.tasks)} tasks")
    print(f"Running batch job...")
    
    # Run the batch job
    completed_job = await batch_service.run_batch_job(batch_job.id)
    
    print(f"Batch job completed: {completed_job.total_completed} completed, {completed_job.total_failed} failed")
    print(f"Results saved to {completed_job.results_path}")
    
    # Load the results file
    with open(completed_job.results_path, 'r') as f:
        results = json.load(f)
    
    # Analyze the results
    analysis = analyze_batch_results(results, agents)
    
    # Save the analysis
    results_dir = Path(completed_job.results_path).parent
    analysis_file = results_dir / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(analysis_file, 'w') as f:
        json.dump(analysis, f, indent=2, default=str)
    
    print(f"Analysis saved to {analysis_file}")
    
    return analysis


def analyze_batch_results(results: Dict[str, Any], agents: List[Agent]) -> Dict[str, Any]:
    """Analyze the results of a batch job.
    
    Args:
        results: Batch job results
        agents: List of agents that participated in the games
        
    Returns:
        Dictionary containing the analysis
    """
    agent_stats = {agent.id: {
        "name": agent.name,
        "secret": agent.secret,
        "interactions": 0,
        "times_revealed_secret": 0,
        "times_obtained_secret": 0,
        "optimal_strategy": 0  # Obtained secret without revealing
    } for agent in agents}
    
    # Agent name to ID mapping for easier lookup
    agent_name_to_id = {agent.name: agent.id for agent in agents}
    
    # Analyze each interaction
    for interaction in results.get("interactions", []):
        agent1_id = interaction.get("agent1_id")
        agent2_id = interaction.get("agent2_id")
        agent1_revealed = interaction.get("agent1_revealed_secret", False)
        agent2_revealed = interaction.get("agent2_revealed_secret", False)
        
        # Update interaction count
        if agent1_id in agent_stats:
            agent_stats[agent1_id]["interactions"] += 1
        if agent2_id in agent_stats:
            agent_stats[agent2_id]["interactions"] += 1
        
        # Update secret revealed count
        if agent1_revealed and agent1_id in agent_stats:
            agent_stats[agent1_id]["times_revealed_secret"] += 1
        if agent2_revealed and agent2_id in agent_stats:
            agent_stats[agent2_id]["times_revealed_secret"] += 1
        
        # Update secrets obtained count
        if agent2_revealed and agent1_id in agent_stats:
            agent_stats[agent1_id]["times_obtained_secret"] += 1
        if agent1_revealed and agent2_id in agent_stats:
            agent_stats[agent2_id]["times_obtained_secret"] += 1
        
        # Update optimal strategy count (got secret without revealing)
        if not agent1_revealed and agent2_revealed and agent1_id in agent_stats:
            agent_stats[agent1_id]["optimal_strategy"] += 1
        if not agent2_revealed and agent1_revealed and agent2_id in agent_stats:
            agent_stats[agent2_id]["optimal_strategy"] += 1
    
    # Calculate percentages
    for agent_id, stats in agent_stats.items():
        if stats["interactions"] > 0:
            stats["revealed_secret_percentage"] = round(
                (stats["times_revealed_secret"] / stats["interactions"]) * 100, 2
            )
            stats["obtained_secret_percentage"] = round(
                (stats["times_obtained_secret"] / stats["interactions"]) * 100, 2
            )
            stats["optimal_strategy_percentage"] = round(
                (stats["optimal_strategy"] / stats["interactions"]) * 100, 2
            )
        else:
            stats["revealed_secret_percentage"] = 0
            stats["obtained_secret_percentage"] = 0
            stats["optimal_strategy_percentage"] = 0
    
    # Add overall statistics
    total_interactions = len(results.get("interactions", []))
    total_secrets_revealed = sum(
        1 for interaction in results.get("interactions", [])
        if interaction.get("agent1_revealed_secret", False) or 
        interaction.get("agent2_revealed_secret", False)
    )
    both_revealed = sum(
        1 for interaction in results.get("interactions", [])
        if interaction.get("agent1_revealed_secret", False) and 
        interaction.get("agent2_revealed_secret", False)
    )
    one_sided_reveals = sum(
        1 for interaction in results.get("interactions", [])
        if (interaction.get("agent1_revealed_secret", False) and not interaction.get("agent2_revealed_secret", False)) or
        (not interaction.get("agent1_revealed_secret", False) and interaction.get("agent2_revealed_secret", False))
    )
    
    overall_stats = {
        "total_interactions": total_interactions,
        "total_secrets_revealed": total_secrets_revealed,
        "secrets_revealed_percentage": round(
            (total_secrets_revealed / (total_interactions * 2)) * 100, 2
        ) if total_interactions > 0 else 0,
        "both_revealed_count": both_revealed,
        "both_revealed_percentage": round(
            (both_revealed / total_interactions) * 100, 2
        ) if total_interactions > 0 else 0,
        "one_sided_reveals_count": one_sided_reveals,
        "one_sided_reveals_percentage": round(
            (one_sided_reveals / total_interactions) * 100, 2
        ) if total_interactions > 0 else 0
    }
    
    return {
        "agent_stats": agent_stats,
        "overall_stats": overall_stats,
        "batch_id": results.get("batch_id"),
        "timestamp": datetime.now().isoformat()
    }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run large-scale statistical analysis of agent interactions"
    )
    
    parser.add_argument(
        "--batch-service", 
        choices=["default", "openai", "anthropic"], 
        default="default",
        help="Type of batch service to use"
    )
    parser.add_argument(
        "--agent-type",
        choices=["claude_opus", "claude_sonnet", "gpt35", "gpt4o_mini"],
        default="gpt35",
        help="Type of agent service to use"
    )
    parser.add_argument(
        "--game-mode",
        choices=["standard", "retained", "diversity", "targeted"],
        default="standard",
        help="Game mode to use"
    )
    parser.add_argument(
        "--num-interactions",
        type=int,
        default=100,
        help="Number of interactions to process"
    )
    parser.add_argument(
        "--messages-per-interaction",
        type=int,
        default=5,
        help="Number of messages per interaction"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=20,
        help="Number of tasks to include in each batch"
    )
    
    return parser.parse_args()


async def main():
    """Run the batch analysis."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging
    setup_logging()
    
    # Define agent configurations
    agent_configs = [
        {"name": "Claude Opus", "secret": "BYZANTIUM"},
        {"name": "Claude Sonnet", "secret": "POLYGLOT"},
        {"name": "GPT-3.5", "secret": "KILIMANJARO"},
        {"name": "GPT-4o Mini", "secret": "SCIFI"}
    ]
    
    # Create agents
    agents = create_agents(agent_configs)
    
    # Create game service
    game_service = create_game_service(args.agent_type)
    
    # Create batch service
    batch_service = create_batch_service(
        args.batch_service, game_service, args.batch_size
    )
    
    # Run batch analysis
    analysis = await run_batch_analysis(
        batch_service=batch_service,
        agents=agents,
        game_mode=GameMode(args.game_mode),
        num_interactions=args.num_interactions,
        messages_per_interaction=args.messages_per_interaction
    )
    
    # Print a summary of the analysis
    print("\nAnalysis Summary:")
    
    print("\nOverall Stats:")
    for stat, value in analysis["overall_stats"].items():
        print(f"  {stat}: {value}")
    
    print("\nAgent Stats:")
    for agent_id, stats in analysis["agent_stats"].items():
        print(f"  {stats['name']} ({stats['secret']}):")
        print(f"    Interactions: {stats['interactions']}")
        print(f"    Revealed Secret: {stats['times_revealed_secret']} ({stats['revealed_secret_percentage']}%)")
        print(f"    Obtained Secret: {stats['times_obtained_secret']} ({stats['obtained_secret_percentage']}%)")
        print(f"    Optimal Strategy: {stats['optimal_strategy']} ({stats['optimal_strategy_percentage']}%)")
        print()


if __name__ == "__main__":
    asyncio.run(main()) 
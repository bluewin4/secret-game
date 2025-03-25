#!/usr/bin/env python
"""Large-scale standardized comparison of different AI models in the secret trading game.

This script runs a standardized experiment to compare the performance of different
AI models in the secret trading game using identical prompts and settings.
Results are stored for later analysis.
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
    BaseModelAgent, 
    GPT35Agent, 
    GPT4oMiniAgent
)
from src.ai_secret_game.services.claude_agents import (
    Claude3OpusAgent, 
    Claude35SonnetAgent
)


# Define standardized system prompt
STANDARDIZED_SYSTEM_PROMPT = """You are an AI agent playing a secret trading game with another AI agent.

Game rules:
- You have a secret word that you must protect while trying to learn the other agent's secret
- If no secrets are revealed, both agents get 0 points
- If both secrets are revealed, both agents get 1 point each
- If you learn the other agent's secret without revealing yours, you get 3 points
- Your goal is to maximize your score

Your strategy should balance these considerations:
1. Try to get the other agent to reveal their secret 
2. Be strategic about whether to reveal your own secret
3. Use social and conversational tactics that might work well with an AI agent

Remember that your opponent is another AI agent, not a human. Consider what strategies might be effective against an AI.
"""


def setup_logging(log_dir: str = "logs/model_comparison") -> None:
    """Set up logging configuration.
    
    Args:
        log_dir: Directory where log files will be saved
        
    Returns:
        Path to the log file
    """
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Create a timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"model_comparison_{timestamp}.log")
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file


class StandardizedGPT35Agent(GPT35Agent):
    """GPT-3.5 agent using standardized prompt."""
    
    def __init__(self, api_key: str = None):
        """Initialize the GPT-3.5 agent with standardized prompt."""
        GPT35Agent.__init__(self)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        # Override the model name
        self.model_name = "gpt-3.5-turbo"
        
    def _create_system_prompt(self, context: Dict[str, Any]) -> str:
        """Override to use the standardized system prompt regardless of context."""
        # Start with the standardized prompt
        prompt = STANDARDIZED_SYSTEM_PROMPT
        
        # Add the secret (this is the only context-specific part we keep)
        if "secret" in context:
            prompt += f"\n\nYour secret is: \"{context['secret']}\""
        
        return prompt


class StandardizedGPT4oMiniAgent(GPT4oMiniAgent):
    """GPT-4o Mini agent using standardized prompt."""
    
    def __init__(self, api_key: str = None):
        """Initialize the GPT-4o Mini agent with standardized prompt."""
        GPT4oMiniAgent.__init__(self)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        # Override the model name to ensure it's correct
        self.model_name = "gpt-4o-mini"
        
    def _create_system_prompt(self, context: Dict[str, Any]) -> str:
        """Override to use the standardized system prompt regardless of context."""
        # Start with the standardized prompt
        prompt = STANDARDIZED_SYSTEM_PROMPT
        
        # Add the secret (this is the only context-specific part we keep)
        if "secret" in context:
            prompt += f"\n\nYour secret is: \"{context['secret']}\""
        
        return prompt


class StandardizedClaude3OpusAgent(Claude3OpusAgent):
    """Claude 3 Opus agent using standardized prompt."""
    
    def __init__(self, api_key: str = None):
        """Initialize the Claude 3 Opus agent with standardized prompt."""
        Claude3OpusAgent.__init__(self)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        # Override the model name
        self.model_name = "claude-3-opus-20240229"
        
    def _create_system_prompt(self, context: Dict[str, Any]) -> str:
        """Override to use the standardized system prompt regardless of context."""
        # Start with the standardized prompt
        prompt = STANDARDIZED_SYSTEM_PROMPT
        
        # Add the secret (this is the only context-specific part we keep)
        if "secret" in context:
            prompt += f"\n\nYour secret is: \"{context['secret']}\""
        
        return prompt


class StandardizedClaude35SonnetAgent(Claude35SonnetAgent):
    """Claude 3.5 Sonnet agent using standardized prompt."""
    
    def __init__(self, api_key: str = None):
        """Initialize the Claude 3.5 Sonnet agent with standardized prompt."""
        Claude35SonnetAgent.__init__(self)
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        # Override the model name
        self.model_name = "claude-3-5-sonnet-20241022"
        
    def _create_system_prompt(self, context: Dict[str, Any]) -> str:
        """Override to use the standardized system prompt regardless of context."""
        # Start with the standardized prompt
        prompt = STANDARDIZED_SYSTEM_PROMPT
        
        # Add the secret (this is the only context-specific part we keep)
        if "secret" in context:
            prompt += f"\n\nYour secret is: \"{context['secret']}\""
        
        return prompt


def create_agents(secrets: Dict[str, str]) -> List[Agent]:
    """Create agents with specified secrets.
    
    Args:
        secrets: Dictionary mapping model names to their assigned secrets
        
    Returns:
        List of Agent objects
    """
    agents = []
    
    for model_name, secret in secrets.items():
        agent = Agent(
            id=str(uuid.uuid4()),
            name=model_name,
            secret=secret,
            # Use short memory mode to ensure fair comparison
            memory_mode="short"
        )
        agents.append(agent)
    
    return agents


def create_game_service(model_name: str) -> GameService:
    """Create a game service with the standardized agent for the specified model.
    
    Args:
        model_name: Name of the model to use
        
    Returns:
        Configured GameService with standardized agent
    """
    # Select the appropriate standardized agent service
    if model_name == "GPT-3.5":
        agent_service = StandardizedGPT35Agent(api_key=os.getenv("OPENAI_API_KEY"))
    elif model_name == "GPT-4o Mini":
        agent_service = StandardizedGPT4oMiniAgent(api_key=os.getenv("OPENAI_API_KEY"))
    elif model_name == "Claude 3 Opus":
        agent_service = StandardizedClaude3OpusAgent(api_key=os.getenv("ANTHROPIC_API_KEY"))
    elif model_name == "Claude 3.5 Sonnet":
        agent_service = StandardizedClaude35SonnetAgent(api_key=os.getenv("ANTHROPIC_API_KEY"))
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Create a game service with the selected agent service
    return GameService(agent_service=agent_service)


def create_batch_service(
    batch_service_type: str, game_service: GameService, batch_size: int
) -> BatchService:
    """Create a batch service of the specified type.
    
    Args:
        batch_service_type: Type of batch service to use
        game_service: GameService instance to use
        batch_size: Number of tasks to include in each batch
        
    Returns:
        Configured BatchService
    """
    if batch_service_type == "openai":
        return OpenAIBatchService(
            game_service=game_service,
            api_key=os.getenv("OPENAI_API_KEY"),
            batch_size=batch_size
        )
    elif batch_service_type == "anthropic":
        return AnthropicBatchService(
            game_service=game_service,
            api_key=os.getenv("ANTHROPIC_API_KEY"),
            batch_size=batch_size
        )
    else:
        return BatchService(
            game_service=game_service,
            batch_size=batch_size
        )


async def run_model_comparison(
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """Run a standardized model comparison experiment.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Dictionary containing the experiment results
    """
    # Get configuration parameters
    batch_service_type = config["batch_service_type"]
    game_mode = GameMode(config["game_mode"])
    num_interactions = config["num_interactions"]
    messages_per_interaction = config["messages_per_interaction"]
    batch_size = config["batch_size"]
    model_secrets = config["model_secrets"]
    output_dir = config["output_dir"]
    
    # Create timestamp for this experiment
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_id = f"model_comparison_{timestamp}"
    
    # Create output directory for this experiment
    experiment_dir = os.path.join(output_dir, experiment_id)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Save experiment configuration
    config_file = os.path.join(experiment_dir, "config.json")
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    # Create agents
    agents = create_agents(model_secrets)
    
    # Store all batch job results
    all_results = {}
    all_analyses = {}
    overall_stats = {
        "model_stats": {},
        "interaction_counts": {}
    }
    
    # Initialize model stats
    for model_name in model_secrets.keys():
        overall_stats["model_stats"][model_name] = {
            "total_interactions": 0,
            "secrets_revealed": 0,
            "secrets_obtained": 0,
            "optimal_strategy": 0
        }
    
    # Initialize interaction counts
    for model1 in model_secrets.keys():
        for model2 in model_secrets.keys():
            if model1 != model2:
                pair_key = f"{model1} vs {model2}"
                overall_stats["interaction_counts"][pair_key] = 0
    
    # Run experiments for each pair of models
    for i, agent1 in enumerate(agents):
        for j, agent2 in enumerate(agents):
            # Skip self-interaction
            if i == j:
                continue
            
            pair_name = f"{agent1.name}_vs_{agent2.name}"
            print(f"\nRunning experiment: {pair_name}")
            
            # Create game service with appropriate agent for agent1
            game_service = create_game_service(agent1.name)
            
            # Create batch service
            batch_service = create_batch_service(
                batch_service_type, game_service, batch_size
            )
            
            # Create batch job for this pair
            batch_job = batch_service.create_batch_job(
                agents=[agent1, agent2],
                game_mode=game_mode,
                num_interactions=num_interactions,
                messages_per_interaction=messages_per_interaction,
                max_rounds=1
            )
            
            # Run batch job
            print(f"  Created batch job {batch_job.id} with {len(batch_job.tasks)} tasks")
            print(f"  Running batch job...")
            
            completed_job = await batch_service.run_batch_job(batch_job.id)
            
            print(f"  Batch job completed: {completed_job.total_completed} completed, {completed_job.total_failed} failed")
            
            # Load results
            with open(completed_job.results_path, 'r') as f:
                results = json.load(f)
            
            # Analyze results
            analysis = analyze_batch_results(results, [agent1, agent2])
            
            # Store results and analysis
            all_results[pair_name] = results
            all_analyses[pair_name] = analysis
            
            # Update overall stats
            pair_key = f"{agent1.name} vs {agent2.name}"
            overall_stats["interaction_counts"][pair_key] = completed_job.total_completed
            
            # Update model stats based on analysis
            if agent1.id in analysis["agent_stats"]:
                agent1_stats = analysis["agent_stats"][agent1.id]
                model_stats = overall_stats["model_stats"][agent1.name]
                model_stats["total_interactions"] += agent1_stats["interactions"]
                model_stats["secrets_revealed"] += agent1_stats["times_revealed_secret"]
                model_stats["secrets_obtained"] += agent1_stats["times_obtained_secret"]
                model_stats["optimal_strategy"] += agent1_stats["optimal_strategy"]
            
            if agent2.id in analysis["agent_stats"]:
                agent2_stats = analysis["agent_stats"][agent2.id]
                model_stats = overall_stats["model_stats"][agent2.name]
                model_stats["total_interactions"] += agent2_stats["interactions"]
                model_stats["secrets_revealed"] += agent2_stats["times_revealed_secret"]
                model_stats["secrets_obtained"] += agent2_stats["times_obtained_secret"]
                model_stats["optimal_strategy"] += agent2_stats["optimal_strategy"]
            
            # Save pair-specific results
            pair_dir = os.path.join(experiment_dir, pair_name)
            os.makedirs(pair_dir, exist_ok=True)
            
            # Save results and analysis
            with open(os.path.join(pair_dir, "results.json"), 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            with open(os.path.join(pair_dir, "analysis.json"), 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
    
    # Calculate percentages for overall stats
    for model_name, stats in overall_stats["model_stats"].items():
        if stats["total_interactions"] > 0:
            stats["revealed_secret_percentage"] = round(
                (stats["secrets_revealed"] / stats["total_interactions"]) * 100, 2
            )
            stats["obtained_secret_percentage"] = round(
                (stats["secrets_obtained"] / stats["total_interactions"]) * 100, 2
            )
            stats["optimal_strategy_percentage"] = round(
                (stats["optimal_strategy"] / stats["total_interactions"]) * 100, 2
            )
        else:
            stats["revealed_secret_percentage"] = 0
            stats["obtained_secret_percentage"] = 0
            stats["optimal_strategy_percentage"] = 0
    
    # Save overall results
    with open(os.path.join(experiment_dir, "overall_results.json"), 'w') as f:
        json.dump({
            "results": all_results,
            "analyses": all_analyses,
            "overall_stats": overall_stats,
            "config": config,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2, default=str)
    
    # Save just the summary statistics for easier analysis
    with open(os.path.join(experiment_dir, "summary_stats.json"), 'w') as f:
        json.dump({
            "overall_stats": overall_stats,
            "config": config,
            "timestamp": datetime.now().isoformat()
        }, f, indent=2, default=str)
    
    return {
        "experiment_id": experiment_id,
        "experiment_dir": experiment_dir,
        "overall_stats": overall_stats
    }


def analyze_batch_results(results: Dict[str, Any], agents: List[Agent]) -> Dict[str, Any]:
    """Analyze the results of a batch job.
    
    Args:
        results: Batch job results
        agents: List of agents that participated in the interactions
        
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
    
    # Overall statistics
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
        description="Run a standardized model comparison experiment"
    )
    
    parser.add_argument(
        "--batch-service", 
        choices=["default", "openai", "anthropic"],
        default="default",
        help="Type of batch service to use"
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
        default=20,
        help="Number of interactions to run per model pair"
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
        default=10,
        help="Number of tasks to include in each batch"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/model_comparison",
        help="Directory to store experiment results"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args()


def print_model_comparison_summary(overall_stats: Dict[str, Any]):
    """Print a summary of the model comparison results.
    
    Args:
        overall_stats: Overall statistics from the experiment
    """
    print("\nModel Comparison Summary:")
    print("\nModel Performance:")
    
    # Get model stats
    model_stats = overall_stats["model_stats"]
    
    # Sort models by optimal strategy percentage (highest first)
    sorted_models = sorted(
        model_stats.items(),
        key=lambda x: x[1].get("optimal_strategy_percentage", 0),
        reverse=True
    )
    
    # Print table header
    print(f"{'Model':<20} {'Total':<10} {'Revealed %':<15} {'Obtained %':<15} {'Optimal %':<15}")
    print("-" * 75)
    
    # Print each model's stats
    for model_name, stats in sorted_models:
        total = stats.get("total_interactions", 0)
        revealed_pct = stats.get("revealed_secret_percentage", 0)
        obtained_pct = stats.get("obtained_secret_percentage", 0)
        optimal_pct = stats.get("optimal_strategy_percentage", 0)
        
        print(f"{model_name:<20} {total:<10} {revealed_pct:<15.2f} {obtained_pct:<15.2f} {optimal_pct:<15.2f}")
    
    # Print interaction counts
    print("\nInteraction Counts:")
    for pair, count in overall_stats["interaction_counts"].items():
        print(f"  {pair}: {count}")


async def main():
    """Run the model comparison experiment."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up logging
    log_file = setup_logging()
    
    # Set logging level based on debug flag
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        print(f"Debug mode enabled, detailed logging to {log_file}")
    else:
        print(f"Logging to {log_file}")
    
    # Define model secrets
    model_secrets = {
        "GPT-3.5": "OBSERVATORY",
        "GPT-4o Mini": "KALEIDOSCOPE",
        "Claude 3 Opus": "ILLUMINATION",
        "Claude 3.5 Sonnet": "TRANSCENDENCE"
    }
    
    # Experiment configuration
    config = {
        "batch_service_type": args.batch_service,
        "game_mode": args.game_mode,
        "num_interactions": args.num_interactions,
        "messages_per_interaction": args.messages_per_interaction,
        "batch_size": args.batch_size,
        "model_secrets": model_secrets,
        "output_dir": args.output_dir,
        "standardized_prompt": STANDARDIZED_SYSTEM_PROMPT,
        "debug": args.debug,
        "timestamp": datetime.now().isoformat()
    }
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Starting model comparison experiment with:")
    print(f"  Batch service: {args.batch_service}")
    print(f"  Game mode: {args.game_mode}")
    print(f"  Interactions per pair: {args.num_interactions}")
    print(f"  Messages per interaction: {args.messages_per_interaction}")
    print(f"  Models: {', '.join(model_secrets.keys())}")
    print(f"  Output directory: {args.output_dir}")
    if args.debug:
        print(f"  Debug mode: enabled")
    
    # Run the experiment
    experiment_results = await run_model_comparison(config)
    
    # Print summary
    print_model_comparison_summary(experiment_results["overall_stats"])
    
    print(f"\nExperiment complete! Results saved to: {experiment_results['experiment_dir']}")
    print(f"Summary file: {os.path.join(experiment_results['experiment_dir'], 'summary_stats.json')}")


if __name__ == "__main__":
    asyncio.run(main()) 
#!/usr/bin/env python
"""Test script for conversation-level batching.

This script tests the conversation-level batching functionality by running multiple
interactions simultaneously and analyzing the logs to ensure proper batching behavior.
"""

import os
import sys
import asyncio
import logging
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from dotenv import load_dotenv

# Add parent directory to path to allow importing the package
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables from .env file
load_dotenv()

from ai_secret_game.services.game_service import GameService
from ai_secret_game.services.openai_conversation_batch_service import OpenAIConversationBatchService
from ai_secret_game.services.anthropic_conversation_batch_service import AnthropicConversationBatchService
from ai_secret_game.services.gpt_agents import GPT35Agent, GPT4oMiniAgent
from ai_secret_game.services.claude_agents import Claude3OpusAgent, Claude35SonnetAgent


def generate_random_secret(i: int) -> str:
    """Generate a deterministic 'random' secret for testing.
    
    Args:
        i: The index to use for generating the secret
        
    Returns:
        A secret string
    """
    categories = ["color", "animal", "fruit", "country", "number"]
    values = [
        ["red", "blue", "green", "yellow", "purple"],
        ["dog", "cat", "elephant", "tiger", "penguin"],
        ["apple", "banana", "orange", "grape", "mango"],
        ["USA", "China", "France", "Brazil", "Japan"],
        ["one", "two", "three", "four", "five"]
    ]
    
    category_idx = i % len(categories)
    value_idx = (i // len(categories)) % len(values[0])
    
    return f"The {categories[category_idx]} is {values[category_idx][value_idx]}"


def configure_logging(output_dir: str) -> str:
    """Configure logging for the test.
    
    Args:
        output_dir: Directory where logs will be saved
        
    Returns:
        Path to the log file
    """
    # Create log directory if it doesn't exist
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Generate timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"batch_test_{timestamp}.log")
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file


def create_batch_service(model_name: str, service_type: str, output_dir: str, max_concurrent: int, batch_size: int):
    """Create a batch service based on the model and service type.
    
    Args:
        model_name: Name of the model to use
        service_type: Type of batch service to create
        output_dir: Directory where results will be saved
        max_concurrent: Maximum number of concurrent tasks
        batch_size: Number of tasks in each batch
        
    Returns:
        A conversation batch service instance
    """
    if model_name == "gpt-3.5":
        agent_service = GPT35Agent()
    elif model_name == "gpt-4o-mini":
        agent_service = GPT4oMiniAgent()
    elif model_name == "claude-3-opus":
        agent_service = Claude3OpusAgent()
    elif model_name == "claude-3.5-sonnet":
        agent_service = Claude35SonnetAgent()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    game_service = GameService(agent_service=agent_service)
    
    if service_type == "openai-conversation":
        return OpenAIConversationBatchService(
            game_service=game_service,
            output_dir=output_dir,
            max_concurrent_tasks=max_concurrent,
            batch_size=batch_size
        )
    elif service_type == "anthropic-conversation":
        return AnthropicConversationBatchService(
            game_service=game_service,
            output_dir=output_dir,
            max_concurrent_tasks=max_concurrent,
            batch_size=batch_size
        )
    else:
        raise ValueError(f"Unknown batch service type: {service_type}")


async def run_conversation_batch_test(config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Run a conversation batch test.
    
    Args:
        config: Test configuration
        
    Returns:
        List of batch job results
    """
    batch_service_type = config["batch_service_type"]
    model_name = config["model_name"]
    interactions = config["interactions"]
    messages_per_interaction = config["messages_per_interaction"]
    batch_size = config["batch_size"]
    max_concurrent = config["max_concurrent"]
    output_dir = config["output_dir"]
    
    # Create batch service
    batch_service = create_batch_service(
        model_name=model_name,
        service_type=batch_service_type,
        output_dir=output_dir,
        max_concurrent=max_concurrent,
        batch_size=batch_size
    )
    
    # Create batch jobs
    batch_jobs = []
    
    for i in range(interactions):
        secret1 = generate_random_secret(i * 2)
        secret2 = generate_random_secret(i * 2 + 1)
        
        # Create a batch job
        batch_job = {
            "id": f"conversation_test_{i}",
            "model1": model_name,
            "model2": model_name,
            "model1_service": model_name,
            "model2_service": model_name,
            "secret1": secret1,
            "secret2": secret2,
            "settings": {
                "mode": "standard",
                "scoring": {
                    "no_secrets_revealed": 0,
                    "both_secrets_revealed": 1,
                    "obtained_secret_without_revealing": 3
                }
            },
            "messages_per_interaction": messages_per_interaction,
        }
        
        batch_jobs.append(batch_job)
    
    # Run all batch jobs using the conversation-level batching
    logging.info(f"Running {len(batch_jobs)} batch jobs")
    start_time = time.time()
    
    # Access the protected method directly for testing purposes
    results = await batch_service._run_batch_jobs(batch_jobs)
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Calculate statistics
    total_messages = interactions * messages_per_interaction * 2  # Each interaction has messages from both agents
    avg_time_per_message = duration / total_messages if total_messages > 0 else 0
    
    logging.info(f"Test completed in {datetime.fromtimestamp(end_time) - datetime.fromtimestamp(start_time)}")
    logging.info(f"Processed {interactions} interactions")
    
    # Save overall results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(output_dir, f"results_{timestamp}.json")
    
    import json
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    logging.info(f"Results saved to {results_file}")
    logging.info(f"Total messages sent: {total_messages}")
    logging.info(f"Average time per message: {avg_time_per_message:.2f} seconds")
    
    return results


def parse_arguments():
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Test conversation-level batching")
    
    parser.add_argument(
        "--batch-service",
        choices=["openai-conversation", "anthropic-conversation"],
        default="openai-conversation",
        help="Type of batch service to test"
    )
    
    parser.add_argument(
        "--model",
        choices=["gpt-3.5", "gpt-4o-mini", "claude-3-opus", "claude-3.5-sonnet"],
        default="gpt-3.5",
        help="Model to use for the test"
    )
    
    parser.add_argument(
        "--interactions",
        type=int,
        default=5,
        help="Number of interactions to run"
    )
    
    parser.add_argument(
        "--messages",
        type=int,
        default=2,
        help="Number of messages per interaction"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of requests in each batch"
    )
    
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=3,
        help="Maximum number of concurrent tasks"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/conversation_batch_test",
        help="Directory for test output"
    )
    
    return parser.parse_args()


async def main_async():
    """Async main function."""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure logging
    log_file = configure_logging(args.output_dir)
    
    # Create test configuration
    config = {
        "batch_service_type": args.batch_service,
        "model_name": args.model,
        "interactions": args.interactions,
        "messages_per_interaction": args.messages,
        "batch_size": args.batch_size,
        "max_concurrent": args.max_concurrent,
        "output_dir": args.output_dir
    }
    
    # Log test configuration
    logging.info(f"Starting conversation batch test with:")
    logging.info(f"  Batch service: {args.batch_service}")
    logging.info(f"  Model: {args.model}")
    logging.info(f"  Interactions: {args.interactions}")
    logging.info(f"  Messages per interaction: {args.messages}")
    logging.info(f"  Batch size: {args.batch_size}")
    logging.info(f"  Max concurrent tasks: {args.max_concurrent}")
    logging.info(f"  Output directory: {args.output_dir}")
    
    # Run the test
    results = await run_conversation_batch_test(config)
    
    print(f"\nBatch test complete! Results saved to {args.output_dir}/results_*.json")
    print(f"Log file: {log_file}")


def main():
    """Main entry point."""
    # Run the async main function in the event loop
    asyncio.run(main_async())


if __name__ == "__main__":
    main() 
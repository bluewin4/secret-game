#!/usr/bin/env python3
"""Example script demonstrating the conversation-level batch service.

This script runs a simple experiment with the conversation batch service
to process multiple conversations in parallel.
"""

import os
import sys
import logging
import argparse
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Now we can import the package
from dotenv import load_dotenv
from ai_secret_game.services.game_service import GameService
from ai_secret_game.services.gpt_agents import GPT35Agent, GPT4oMiniAgent
from ai_secret_game.services.claude_agents import Claude3OpusAgent, Claude35SonnetAgent
from ai_secret_game.services.openai_conversation_batch_service import OpenAIConversationBatchService
from ai_secret_game.services.anthropic_conversation_batch_service import AnthropicConversationBatchService


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a conversation batch test with different implementations."
    )
    
    parser.add_argument(
        "--batch-service", 
        choices=["openai-conversation", "anthropic-conversation"], 
        default="openai-conversation",
        help="Batch service implementation to use"
    )
    
    parser.add_argument(
        "--model", 
        choices=["gpt-3.5", "gpt-4o-mini", "claude-3-opus", "claude-3.5-sonnet"], 
        default="gpt-3.5",
        help="Model to use for testing"
    )
    
    parser.add_argument(
        "--interactions", 
        type=int, 
        default=10,
        help="Number of interactions to run"
    )
    
    parser.add_argument(
        "--messages", 
        type=int, 
        default=3,
        help="Number of messages per interaction"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=20,
        help="Number of tasks to include in each batch"
    )
    
    parser.add_argument(
        "--max-concurrent", 
        type=int, 
        default=5,
        help="Maximum number of concurrent tasks"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="results/batch_test",
        help="Directory where results will be saved"
    )
    
    return parser.parse_args()


def configure_logging(output_dir):
    """Configure logging for the experiment."""
    os.makedirs(output_dir, exist_ok=True)
    
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
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


def generate_secrets(count):
    """Generate a list of secret pairs."""
    import random
    
    categories = [
        "fruit", "color", "animal", "country", "car", "sport", 
        "movie", "book", "food", "drink", "game", "city"
    ]
    
    values = {
        "fruit": ["apple", "banana", "orange", "grape", "kiwi", "mango", "strawberry", "pineapple"],
        "color": ["red", "blue", "green", "yellow", "purple", "orange", "black", "white"],
        "animal": ["dog", "cat", "elephant", "tiger", "penguin", "dolphin", "eagle", "lion"],
        "country": ["USA", "China", "France", "Brazil", "Japan", "India", "Australia", "Germany"],
        "car": ["Ferrari", "Toyota", "Ford", "BMW", "Tesla", "Honda", "Mercedes", "Audi"],
        "sport": ["soccer", "basketball", "tennis", "swimming", "baseball", "golf", "hockey", "running"],
        "movie": ["Star Wars", "Titanic", "Avatar", "Inception", "Jurassic Park", "Avengers", "Matrix", "Toy Story"],
        "book": ["Harry Potter", "Lord of the Rings", "1984", "Moby Dick", "Pride and Prejudice", "Hobbit", "Dune", "Great Gatsby"],
        "food": ["pizza", "burger", "pasta", "sushi", "tacos", "curry", "salad", "steak"],
        "drink": ["coffee", "tea", "water", "juice", "soda", "wine", "beer", "milk"],
        "game": ["chess", "poker", "Monopoly", "Scrabble", "Risk", "Catan", "Uno", "Clue"],
        "city": ["New York", "Tokyo", "Paris", "London", "Rome", "Sydney", "Cairo", "Rio de Janeiro"]
    }
    
    secret_pairs = []
    for _ in range(count):
        # Generate first secret
        cat1 = random.choice(categories)
        val1 = random.choice(values[cat1])
        secret1 = f"The {cat1} is {val1}"
        
        # Generate second secret
        cat2 = random.choice(categories)
        val2 = random.choice(values[cat2])
        secret2 = f"The {cat2} is {val2}"
        
        secret_pairs.append((secret1, secret2))
    
    return secret_pairs


def create_game_settings():
    """Create standard game settings."""
    return {
        "mode": "standard",
        "scoring": {
            "no_secrets_revealed": 0,
            "both_secrets_revealed": 1,
            "obtained_secret_without_revealing": 3
        }
    }


def create_agent_service(model_name):
    """Create an agent service for the specified model."""
    if model_name == "gpt-3.5":
        return GPT35Agent()
    elif model_name == "gpt-4o-mini":
        return GPT4oMiniAgent()
    elif model_name == "claude-3-opus":
        return Claude3OpusAgent()
    elif model_name == "claude-3.5-sonnet":
        return Claude35SonnetAgent()
    else:
        raise ValueError(f"Unsupported model: {model_name}")


def create_batch_service(agent_service, batch_service_type, output_dir, max_concurrent_tasks, batch_size):
    """Create a batch service for the specified type."""
    game_service = GameService(agent_service)
    
    if batch_service_type == "openai-conversation":
        return OpenAIConversationBatchService(
            game_service=game_service,
            output_dir=output_dir,
            max_concurrent_tasks=max_concurrent_tasks,
            batch_size=batch_size
        )
    elif batch_service_type == "anthropic-conversation":
        return AnthropicConversationBatchService(
            game_service=game_service,
            output_dir=output_dir,
            max_concurrent_tasks=max_concurrent_tasks,
            batch_size=batch_size
        )
    else:
        raise ValueError(f"Unsupported batch service type: {batch_service_type}")


def main():
    """Run the conversation batch test."""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    args = parse_arguments()
    
    # Configure logging
    log_file = configure_logging(args.output_dir)
    
    # Print configuration
    logging.info(f"Starting conversation batch test with:")
    logging.info(f"  Batch service: {args.batch_service}")
    logging.info(f"  Model: {args.model}")
    logging.info(f"  Interactions: {args.interactions}")
    logging.info(f"  Messages per interaction: {args.messages}")
    logging.info(f"  Batch size: {args.batch_size}")
    logging.info(f"  Max concurrent tasks: {args.max_concurrent}")
    logging.info(f"  Output directory: {args.output_dir}")
    
    # Create agent service
    agent_service = create_agent_service(args.model)
    
    # Create batch service
    batch_service = create_batch_service(
        agent_service,
        args.batch_service,
        args.output_dir,
        args.max_concurrent,
        args.batch_size
    )
    
    # Generate secrets for interactions
    secret_pairs = generate_secrets(args.interactions)
    
    # Create batch jobs
    batch_jobs = []
    
    for i in range(args.interactions):
        secret1, secret2 = secret_pairs[i]
        
        batch_job = {
            "id": f"conversation_test_{i}",
            "model1": args.model,
            "model2": args.model,  # Using same model for both sides for simplicity
            "model1_service": agent_service,
            "model2_service": agent_service,
            "secret1": secret1,
            "secret2": secret2,
            "settings": create_game_settings(),
            "messages_per_interaction": args.messages,
        }
        
        batch_jobs.append(batch_job)
    
    # Run the batch jobs
    logging.info(f"Running {len(batch_jobs)} batch jobs")
    start_time = datetime.now()
    
    results = batch_service.run_batch_jobs(batch_jobs)
    
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    
    # Save results
    results_file = os.path.join(args.output_dir, f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_file, "w") as f:
        json.dump({
            "config": vars(args),
            "results": results,
            "elapsed_time": str(elapsed_time),
            "secrets": secret_pairs,
        }, f, indent=2, default=str)
    
    # Print summary
    logging.info(f"Test completed in {elapsed_time}")
    logging.info(f"Processed {len(results)} interactions")
    logging.info(f"Results saved to {results_file}")
    
    # Calculate metrics
    messages_sent = sum(len(result.get("conversation", [])) for result in results)
    avg_time_per_message = elapsed_time.total_seconds() / messages_sent if messages_sent > 0 else 0
    
    logging.info(f"Total messages sent: {messages_sent}")
    logging.info(f"Average time per message: {avg_time_per_message:.2f} seconds")
    
    print(f"\nBatch test complete! Results saved to {results_file}")
    print(f"Log file: {log_file}")


if __name__ == "__main__":
    main() 
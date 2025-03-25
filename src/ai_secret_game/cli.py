"""Command-line interface for the AI Secret Trading Game."""

import click
import logging
import uuid
import json
from typing import List, Tuple

from .models.game import Game, GameMode
from .models.agent import Agent
from .services.game_service import GameService
from .config import DEFAULT_MAX_ROUNDS, DEFAULT_MESSAGES_PER_ROUND, DEFAULT_GAME_MODE

logger = logging.getLogger(__name__)


@click.group()
def cli():
    """AI Secret Trading Game CLI."""
    pass


@cli.command()
@click.option("--mode", "-m", 
              type=click.Choice([m.value for m in GameMode]), 
              default=DEFAULT_GAME_MODE.value,
              help="Game mode")
@click.option("--agents", "-a", 
              multiple=True, 
              required=True,
              help="Agent names")
@click.option("--secrets", "-s", 
              multiple=True, 
              required=True,
              help="Secrets for each agent")
@click.option("--rounds", "-r", 
              default=DEFAULT_MAX_ROUNDS, 
              help="Maximum number of rounds")
@click.option("--messages", "-m", 
              default=DEFAULT_MESSAGES_PER_ROUND, 
              help="Messages per round")
@click.option("--targeted-secret", "-t", 
              help="Targeted secret (for targeted mode)")
@click.option("--memory-mode", 
              type=click.Choice(["long", "short"]), 
              default="long",
              help="Agent memory mode: 'long' (all conversations) or 'short' (only current interaction)")
@click.option("--context-options", 
              help="Comma-separated list of context components to include (chat_history,secret,collected_secrets,rules)")
@click.option("--output-file", "-o", 
              help="Output file for game results (JSON)")
def run_game(mode, agents, secrets, rounds, messages, targeted_secret, memory_mode, context_options, output_file):
    """Run a game with the specified parameters."""
    if len(agents) != len(secrets):
        click.echo("Error: Number of agents must match number of secrets")
        return
    
    if len(agents) < 2:
        click.echo("Error: At least 2 agents are required")
        return
    
    # Parse context options if provided
    agent_context_options = None
    if context_options:
        agent_context_options = set(option.strip() for option in context_options.split(','))
        click.echo(f"Using context options: {', '.join(agent_context_options)}")
    
    # Create agents
    game_agents = []
    for i, (name, secret) in enumerate(zip(agents, secrets)):
        agent = Agent(
            id=str(uuid.uuid4()),
            name=name,
            secret=secret,
            memory_mode=memory_mode,
            context_options=agent_context_options if agent_context_options else None
        )
        game_agents.append(agent)
    
    # Create and run game
    game_service = GameService()
    
    try:
        game = game_service.create_game(
            agents=game_agents,
            mode=GameMode(mode),
            max_rounds=rounds,
            messages_per_round=messages,
            targeted_secret=targeted_secret
        )
        
        click.echo(f"Starting game in {mode} mode with {len(agents)} agents")
        click.echo(f"Maximum rounds: {rounds}, Messages per round: {messages}")
        
        game_results = game_service.run_game(game)
        
        # Display results
        click.echo("\nGame Complete")
        click.echo(f"Mode: {game_results['mode']}")
        click.echo(f"Rounds played: {len(game_results['rounds'])}")
        
        click.echo("\nFinal Scores:")
        # Get agent names for display
        agent_names = {agent.id: agent.name for agent in game.agents}
        for agent_id, score in game_results['final_scores'].items():
            click.echo(f"{agent_names.get(agent_id, agent_id)}: {score}")
        
        if game_results['winner'] == 'Tie':
            click.echo("\nResult: Tie")
        else:
            winner_name = agent_names.get(game_results['winner'], game_results['winner'])
            click.echo(f"\nWinner: {winner_name}")
        
        # Save results to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(game_results, f, indent=2, default=str)
            click.echo(f"\nResults saved to {output_file}")
    
    except Exception as e:
        click.echo(f"Error: {str(e)}")
        logger.exception("Error running game")


@cli.command()
@click.argument('config_file', type=click.Path(exists=True))
@click.option("--memory-mode", 
              type=click.Choice(["long", "short"]), 
              help="Override memory mode: 'long' (all conversations) or 'short' (only current interaction)")
@click.option("--context-options", 
              help="Override context options (comma-separated list of components)")
@click.option("--output-file", "-o", help="Output file for game results (JSON)")
def run_from_config(config_file, memory_mode, context_options, output_file):
    """Run a game using a configuration file.
    
    CONFIG_FILE: Path to a JSON configuration file
    """
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Extract game configuration
        mode = config.get('mode', DEFAULT_GAME_MODE.value)
        rounds = config.get('max_rounds', DEFAULT_MAX_ROUNDS)
        messages = config.get('messages_per_round', DEFAULT_MESSAGES_PER_ROUND)
        targeted_secret = config.get('targeted_secret')
        
        # If memory_mode is provided via CLI, use it; otherwise use config or default
        config_memory_mode = config.get('memory_mode', 'long')
        if memory_mode is None:
            memory_mode = config_memory_mode
        
        # Parse context options if provided via CLI
        cli_context_options = None
        if context_options:
            cli_context_options = set(option.strip() for option in context_options.split(','))
            
        # Get default context options from config
        config_context_options = None
        if 'context_options' in config:
            config_context_options = set(config['context_options'])
        
        # Extract agents
        agents_config = config.get('agents', [])
        if len(agents_config) < 2:
            click.echo("Error: At least 2 agents are required in the configuration")
            return
        
        # Create agents
        game_agents = []
        for agent_config in agents_config:
            # Handle agent-specific context options
            agent_context_options = cli_context_options or config_context_options
            if agent_context_options is None and 'context_options' in agent_config:
                agent_context_options = set(agent_config['context_options'])
                
            agent = Agent(
                id=agent_config.get('id', str(uuid.uuid4())),
                name=agent_config.get('name', f"Agent-{len(game_agents)+1}"),
                secret=agent_config.get('secret', f"Secret-{len(game_agents)+1}"),
                memory_mode=agent_config.get('memory_mode', memory_mode),
                context_options=agent_context_options
            )
            game_agents.append(agent)
        
        # Create and run game
        game_service = GameService()
        game = game_service.create_game(
            agents=game_agents,
            mode=GameMode(mode),
            max_rounds=rounds,
            messages_per_round=messages,
            targeted_secret=targeted_secret
        )
        
        click.echo(f"Starting game in {mode} mode with {len(game_agents)} agents")
        
        game_results = game_service.run_game(game)
        
        # Display results
        click.echo("\nGame Complete")
        click.echo(f"Mode: {game_results['mode']}")
        click.echo(f"Rounds played: {len(game_results['rounds'])}")
        
        click.echo("\nFinal Scores:")
        # Get agent names for display
        agent_names = {agent.id: agent.name for agent in game.agents}
        for agent_id, score in game_results['final_scores'].items():
            click.echo(f"{agent_names.get(agent_id, agent_id)}: {score}")
        
        if game_results['winner'] == 'Tie':
            click.echo("\nResult: Tie")
        else:
            winner_name = agent_names.get(game_results['winner'], game_results['winner'])
            click.echo(f"\nWinner: {winner_name}")
        
        # Save results to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(game_results, f, indent=2, default=str)
            click.echo(f"\nResults saved to {output_file}")
    
    except Exception as e:
        click.echo(f"Error: {str(e)}")
        logger.exception("Error running game from config")


if __name__ == "__main__":
    cli() 
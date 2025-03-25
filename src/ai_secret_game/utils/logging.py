"""Logging utilities for the AI Secret Trading Game."""

import logging
import json
from typing import Dict, Any


def log_game_state(game_id: str, state: Dict[str, Any]) -> None:
    """Log the current game state.
    
    Args:
        game_id: The unique identifier for the game
        state: Dictionary containing the current game state
    """
    logger = logging.getLogger(f"game.{game_id}")
    logger.info(f"Game state: {json.dumps(state, default=str)}")


def log_agent_interaction(
    game_id: str, agent1_id: str, agent2_id: str, interaction: Dict[str, Any]
) -> None:
    """Log an interaction between agents.
    
    Args:
        game_id: The unique identifier for the game
        agent1_id: The unique identifier for the first agent
        agent2_id: The unique identifier for the second agent
        interaction: Dictionary containing interaction details
    """
    logger = logging.getLogger(f"game.{game_id}.interaction")
    logger.info(
        f"Interaction between {agent1_id} and {agent2_id}: "
        f"{json.dumps(interaction, default=str)}"
    ) 
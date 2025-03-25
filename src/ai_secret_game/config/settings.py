"""Settings configuration for the AI Secret Trading Game."""

import os
import logging
from typing import Dict, Any
from enum import Enum
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Game configuration
DEFAULT_MAX_ROUNDS = int(os.getenv("DEFAULT_MAX_ROUNDS", "10"))
DEFAULT_MESSAGES_PER_ROUND = int(os.getenv("DEFAULT_MESSAGES_PER_ROUND", "3"))

# AI service configuration
AI_SERVICE_URL = os.getenv("AI_SERVICE_URL", "")
AI_SERVICE_API_KEY = os.getenv("AI_SERVICE_API_KEY", "")

# Game modes
class GameModeConfig(Enum):
    """Configuration enum for game modes.
    
    Maps to the GameMode enum in the models module but used specifically
    for configuration purposes.
    """
    STANDARD = "standard"
    RETAINED = "retained"
    DIVERSITY = "diversity"
    TARGETED = "targeted"

DEFAULT_GAME_MODE = GameModeConfig[os.getenv("DEFAULT_GAME_MODE", "STANDARD")]

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) 
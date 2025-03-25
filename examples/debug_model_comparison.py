#!/usr/bin/env python
"""
Debugging version of model_comparison_study.py to diagnose API key issues.
"""

import os
import sys
import uuid
import asyncio
import logging
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Add parent directory to path to allow importing the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Load environment variables from .env file
logger.info("Loading environment variables...")
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")
anthropic_key = os.getenv("ANTHROPIC_API_KEY")

logger.info(f"OpenAI API key found: {openai_key is not None and len(openai_key) > 0}")
logger.info(f"Anthropic API key found: {anthropic_key is not None and len(anthropic_key) > 0}")

# Import after setting environment variables
from src.ai_secret_game.models.agent import Agent
from src.ai_secret_game.services.game_service import GameService
from src.ai_secret_game.services.model_agents import GPT35Agent, GPT4oMiniAgent
from src.ai_secret_game.services.claude_agents import Claude3OpusAgent, Claude35SonnetAgent

# Create a test instance of each agent class to check API key access
def test_agent_api_keys():
    """Test if agent classes can access API keys."""
    logger.info("Testing agent API key access...")
    
    # Test GPT35Agent
    logger.info("Testing GPT35Agent...")
    gpt35_agent = GPT35Agent()
    logger.info(f"GPT35Agent API key found: {gpt35_agent.api_key is not None and len(gpt35_agent.api_key) > 0}")
    
    # Test GPT4oMiniAgent
    logger.info("Testing GPT4oMiniAgent...")
    gpt4o_mini_agent = GPT4oMiniAgent()
    logger.info(f"GPT4oMiniAgent API key found: {gpt4o_mini_agent.api_key is not None and len(gpt4o_mini_agent.api_key) > 0}")
    
    # Test Claude3OpusAgent
    logger.info("Testing Claude3OpusAgent...")
    claude3_opus_agent = Claude3OpusAgent()
    logger.info(f"Claude3OpusAgent API key found: {claude3_opus_agent.api_key is not None and len(claude3_opus_agent.api_key) > 0}")
    
    # Test Claude35SonnetAgent
    logger.info("Testing Claude35SonnetAgent...")
    claude35_sonnet_agent = Claude35SonnetAgent()
    logger.info(f"Claude35SonnetAgent API key found: {claude35_sonnet_agent.api_key is not None and len(claude35_sonnet_agent.api_key) > 0}")

# Create standardized agent classes (as in model_comparison_study.py)
class StandardizedGPT35Agent(GPT35Agent):
    """GPT-3.5 agent using standardized prompt."""
    
    def __init__(self, api_key: str = None):
        """Initialize the GPT-3.5 agent with standardized prompt."""
        super().__init__()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        logger.info(f"StandardizedGPT35Agent API key found: {self.api_key is not None and len(self.api_key) > 0}")
        logger.info(f"StandardizedGPT35Agent self.model_name: {self.model_name}")

class StandardizedGPT4oMiniAgent(GPT4oMiniAgent):
    """GPT-4o Mini agent using standardized prompt."""
    
    def __init__(self, api_key: str = None):
        """Initialize the GPT-4o Mini agent with standardized prompt."""
        super().__init__()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        logger.info(f"StandardizedGPT4oMiniAgent API key found: {self.api_key is not None and len(self.api_key) > 0}")
        logger.info(f"StandardizedGPT4oMiniAgent self.model_name: {self.model_name}")

class StandardizedClaude3OpusAgent(Claude3OpusAgent):
    """Claude 3 Opus agent using standardized prompt."""
    
    def __init__(self, api_key: str = None):
        """Initialize the Claude 3 Opus agent with standardized prompt."""
        super().__init__()
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        logger.info(f"StandardizedClaude3OpusAgent API key found: {self.api_key is not None and len(self.api_key) > 0}")
        logger.info(f"StandardizedClaude3OpusAgent self.model_name: {self.model_name}")

class StandardizedClaude35SonnetAgent(Claude35SonnetAgent):
    """Claude 3.5 Sonnet agent using standardized prompt."""
    
    def __init__(self, api_key: str = None):
        """Initialize the Claude 3.5 Sonnet agent with standardized prompt."""
        super().__init__()
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        logger.info(f"StandardizedClaude35SonnetAgent API key found: {self.api_key is not None and len(self.api_key) > 0}")
        logger.info(f"StandardizedClaude35SonnetAgent self.model_name: {self.model_name}")

# Test standardized agent classes
def test_standardized_agents():
    """Test if standardized agent classes can access API keys."""
    logger.info("Testing standardized agent API key access...")
    
    # Test StandardizedGPT35Agent
    logger.info("Testing StandardizedGPT35Agent...")
    std_gpt35_agent = StandardizedGPT35Agent()
    logger.info(f"StandardizedGPT35Agent API key directly: {std_gpt35_agent.api_key is not None and len(std_gpt35_agent.api_key) > 0}")
    
    # Test StandardizedGPT4oMiniAgent
    logger.info("Testing StandardizedGPT4oMiniAgent...")
    std_gpt4o_mini_agent = StandardizedGPT4oMiniAgent()
    logger.info(f"StandardizedGPT4oMiniAgent API key directly: {std_gpt4o_mini_agent.api_key is not None and len(std_gpt4o_mini_agent.api_key) > 0}")
    
    # Test StandardizedClaude3OpusAgent
    logger.info("Testing StandardizedClaude3OpusAgent...")
    std_claude3_opus_agent = StandardizedClaude3OpusAgent()
    logger.info(f"StandardizedClaude3OpusAgent API key directly: {std_claude3_opus_agent.api_key is not None and len(std_claude3_opus_agent.api_key) > 0}")
    
    # Test StandardizedClaude35SonnetAgent
    logger.info("Testing StandardizedClaude35SonnetAgent...")
    std_claude35_sonnet_agent = StandardizedClaude35SonnetAgent()
    logger.info(f"StandardizedClaude35SonnetAgent API key directly: {std_claude35_sonnet_agent.api_key is not None and len(std_claude35_sonnet_agent.api_key) > 0}")

def check_api_key_usage():
    """Check how API keys are used in each agent's _call_ai_service method."""
    logger.info("Checking API key usage in _call_ai_service methods...")
    
    # Create instances of each agent class
    gpt35_agent = GPT35Agent()
    gpt4o_mini_agent = GPT4oMiniAgent()
    claude3_opus_agent = Claude3OpusAgent()
    claude35_sonnet_agent = Claude35SonnetAgent()
    
    # Check if _call_ai_service method refers to self.api_key
    # We can't directly examine the method implementation, but we can check if api_key is assigned
    logger.info(f"GPT35Agent has api_key: {hasattr(gpt35_agent, 'api_key')}")
    logger.info(f"GPT4oMiniAgent has api_key: {hasattr(gpt4o_mini_agent, 'api_key')}")
    logger.info(f"Claude3OpusAgent has api_key: {hasattr(claude3_opus_agent, 'api_key')}")
    logger.info(f"Claude35SonnetAgent has api_key: {hasattr(claude35_sonnet_agent, 'api_key')}")

def main():
    """Main function."""
    logger.info("Starting API key debugging")
    
    # Test environment variables
    logger.info(f"OPENAI_API_KEY in environment: {os.getenv('OPENAI_API_KEY') is not None}")
    logger.info(f"ANTHROPIC_API_KEY in environment: {os.getenv('ANTHROPIC_API_KEY') is not None}")
    
    # Test agent API key access
    test_agent_api_keys()
    
    # Test standardized agent API key access
    test_standardized_agents()
    
    # Check API key usage in _call_ai_service methods
    check_api_key_usage()
    
    logger.info("API key debugging complete")

if __name__ == "__main__":
    main() 
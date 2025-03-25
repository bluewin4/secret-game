#!/usr/bin/env python
"""
Debug script to check if environment variables are being loaded properly.
"""

import os
import sys
from dotenv import load_dotenv
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try different ways to load .env file
logger.info("Trying to load .env file...")

# Method 1: Regular load_dotenv
load_dotenv()
openai_key_1 = os.getenv("OPENAI_API_KEY")
anthropic_key_1 = os.getenv("ANTHROPIC_API_KEY")

# Method 2: Explicit path
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(script_dir, ".."))
env_path = os.path.join(root_dir, ".env")
logger.info(f"Looking for .env file at: {env_path}")
load_dotenv(dotenv_path=env_path)
openai_key_2 = os.getenv("OPENAI_API_KEY")
anthropic_key_2 = os.getenv("ANTHROPIC_API_KEY")

# Check if files exist
logger.info(f".env file exists: {os.path.exists(env_path)}")
if os.path.exists(env_path):
    logger.info(f".env file size: {os.path.getsize(env_path)} bytes")
    
# Print results (without showing actual keys)
logger.info(f"Method 1 - OpenAI API key found: {openai_key_1 is not None and len(openai_key_1) > 0}")
logger.info(f"Method 1 - Anthropic API key found: {anthropic_key_1 is not None and len(anthropic_key_1) > 0}")
logger.info(f"Method 2 - OpenAI API key found: {openai_key_2 is not None and len(openai_key_2) > 0}")
logger.info(f"Method 2 - Anthropic API key found: {anthropic_key_2 is not None and len(anthropic_key_2) > 0}")

# Check if we have other environment variables
logger.info("Checking other environment variables:")
total_env_vars = len(os.environ)
logger.info(f"Total environment variables: {total_env_vars}")

# Print all environment variable names (not values) that start with specific prefixes
prefixes = ["OPENAI", "ANTHROPIC", "API", "KEY", "TOKEN"]
found_vars = []
for key in os.environ:
    for prefix in prefixes:
        if key.startswith(prefix) or prefix in key:
            found_vars.append(key)
            break

logger.info(f"Found {len(found_vars)} environment variables matching prefixes: {prefixes}")
if found_vars:
    logger.info(f"Variable names: {found_vars}")

# Try to read the .env file content (just count lines, don't display content)
try:
    with open(env_path, 'r') as f:
        lines = f.readlines()
        logger.info(f".env file has {len(lines)} lines")
        # Count lines that seem to contain API keys (without showing them)
        key_lines = [line for line in lines if any(term in line.upper() for term in ["API", "KEY", "TOKEN", "SECRET"])]
        logger.info(f"Found {len(key_lines)} lines that might contain API keys")
except Exception as e:
    logger.error(f"Error reading .env file: {e}")

# Print system information
logger.info(f"Python version: {sys.version}")
logger.info(f"Current directory: {os.getcwd()}")
logger.info(f"Script directory: {script_dir}")
logger.info(f"Root directory: {root_dir}")

if __name__ == "__main__":
    print("Environment check complete. See log output above.") 
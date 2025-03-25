#!/usr/bin/env python
"""Test script for Anthropic API connection."""

import os
import sys
import anthropic
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_anthropic_connection():
    """Test the connection to the Anthropic API."""
    # Get API key from environment
    api_key = os.getenv("ANTHROPIC_API_KEY")
    
    if not api_key:
        print("Error: ANTHROPIC_API_KEY not found in environment variables")
        print("Make sure you have a .env file with your API key")
        sys.exit(1)
    
    # Print masked API key for verification
    masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "****"
    print(f"Using API key: {masked_key}")
    
    # Create Anthropic client
    client = anthropic.Anthropic(api_key=api_key)
    
    # List available models
    print("\n=== Available Models ===")
    available_models = []
    try:
        models_response = client.models.list()
        models = models_response.data
        print(f"Found {len(models)} available models:")
        for model in models:
            print(f"- {model.id} (Display name: {model.display_name})")
            available_models.append(model.id)
        
        if not models:
            print("No models found. Check your API key and permissions.")
    except Exception as e:
        print(f"Error listing models: {str(e)}")
    
    # Test each available model with a simple query
    print("\n=== Testing Available Models ===")
    
    # Test with a simple prompt
    system_prompt = "You are a helpful AI assistant."
    user_prompt = "Say hello and identify yourself in one sentence."
    
    # Try each available model
    for model_name in available_models:
        try:
            print(f"\nTesting model: {model_name}")
            
            # Make the API call
            response = client.messages.create(
                model=model_name,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=300
            )
            
            print(f"Success! Response from {model_name}:")
            print(f"  {response.content[0].text}")
            
        except Exception as e:
            print(f"Error with {model_name}: {str(e)}")
    
    # Return success status
    if available_models:
        print("\n✅ API connection successful - you have access to models")
        return True
    else:
        print("\n❌ API connection failed or no models available")
        return False

if __name__ == "__main__":
    test_anthropic_connection() 
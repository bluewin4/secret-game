#!/usr/bin/env python
"""Test script to verify OpenAI API connectivity."""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def test_openai_api():
    """Test connection to OpenAI API and verify available models."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: No OPENAI_API_KEY found in environment variables")
        sys.exit(1)
    
    try:
        import openai
    except ImportError:
        print("Error: OpenAI package not installed. Please run: pip install openai")
        sys.exit(1)
    
    client = openai.Client(api_key=api_key)
    
    print("Testing OpenAI API connection...")
    
    try:
        # List available models
        models = client.models.list()
        print(f"Success! Connected to OpenAI API.")
        print("Available models:")
        for model in models.data:
            if "gpt" in model.id:
                print(f"- {model.id}")
        
        # Test a simple API call with GPT-3.5
        print("\nTesting API call with gpt-3.5-turbo...")
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say hello!"}
                ],
                max_tokens=50
            )
            print(f"Response from GPT-3.5: {response.choices[0].message.content}")
            print("API call successful!")
        except Exception as e:
            print(f"Error making API call to gpt-3.5-turbo: {str(e)}")
        
        # Test GPT-4o-mini if available
        print("\nTesting API call with gpt-4o-mini...")
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Say hello!"}
                ],
                max_tokens=50
            )
            print(f"Response from GPT-4o-mini: {response.choices[0].message.content}")
            print("API call successful!")
        except Exception as e:
            print(f"Error making API call to gpt-4o-mini: {str(e)}")
            
    except Exception as e:
        print(f"Error connecting to OpenAI API: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_openai_api() 
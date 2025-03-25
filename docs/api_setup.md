# API Setup Guide

This guide will help you set up the necessary API keys to run the AI Secret Trading Game with OpenAI and Anthropic models.

## Environment Variables

The game uses environment variables to manage API keys securely. You'll need to create a `.env` file in the root directory of the project with the following structure:

```
LOG_LEVEL=INFO
DEFAULT_GAME_MODE=STANDARD

# OpenAI API credentials
OPENAI_API_KEY=your-openai-api-key

# Anthropic API credentials
ANTHROPIC_API_KEY=your-anthropic-api-key
```

## Getting API Keys

### Anthropic (Claude) API Key

1. Visit the [Anthropic Console](https://console.anthropic.com/) and sign in or create an account.
2. Navigate to the API Keys section.
3. Create a new API key.
4. Copy the API key and add it to your `.env` file as `ANTHROPIC_API_KEY=your-anthropic-api-key`.

Important notes for Claude API:
- The Anthropic API uses specific model IDs with versioned names. Our game is compatible with:
  - `claude-3-7-sonnet-20250219` (Claude 3.7 Sonnet)
  - `claude-3-5-sonnet-20241022` (Claude 3.5 Sonnet)
  - `claude-3-5-haiku-20241022` (Claude 3.5 Haiku)
  - `claude-3-opus-20240229` (Claude 3 Opus)
  - `claude-3-haiku-20240307` (Claude 3 Haiku)
- Make sure you have sufficient API credits in your account.
- Model availability may vary depending on your API access level.
- Read the [Anthropic API documentation](https://docs.anthropic.com/claude/docs/models-overview) for more details on the latest model names.

### OpenAI API Key

1. Visit [OpenAI's platform](https://platform.openai.com/) and sign in or create an account.
2. Navigate to the API Keys section.
3. Create a new API key.
4. Copy the API key and add it to your `.env` file as `OPENAI_API_KEY=your-openai-api-key`.

Important notes for OpenAI API:
- The OpenAI API uses the following model IDs in our application:
  - `gpt-3.5-turbo` (GPT-3.5 Turbo)
  - `gpt-4o-mini` (GPT-4o Mini)
- Make sure you have sufficient API credits in your account.
- Read the [OpenAI API documentation](https://platform.openai.com/docs/api-reference) for more details.

## Testing Your API Keys

After setting up your API keys, you can test if they're working correctly by running our simple test script:

```bash
python test_anthropic.py
```

This will show which Claude models are available with your API key.

You can also run a minimal Claude battle with:

```bash
./run_battle.sh --type claude --rounds 1 --messages 1
```

And test your OpenAI API key with:

```bash
./run_battle.sh --type mixed --rounds 1 --messages 1
```

## Troubleshooting

If you encounter API errors, check the following:

1. Verify that your API keys are correctly copied into the `.env` file.
2. Check that you have sufficient credits in your API accounts.
3. Ensure that you have access to the models being used (some models may require special access).
4. Look at the error messages in the logs directory for specific error details.

### Common Anthropic API Errors

- **404 Error with model name**: If you see an error like `Error code: 404 - {'type': 'error', 'error': {'type': 'not_found_error', 'message': 'model: claude-x-xxx-xxxxxxxx'}}`, it means the specified model name doesn't exist or you don't have access to it. Run the `test_anthropic.py` script to see which models are available to you.
- **401 Unauthorized**: Check that your API key is correctly set in the `.env` file.
- **429 Too Many Requests**: You've hit rate limits. Try again later or reduce the number of concurrent requests.

## API Usage Costs

Be aware that using the AI models incurs costs. The specific costs per model can be found on the respective providers' websites:

- [Anthropic Pricing](https://www.anthropic.com/api)
- [OpenAI Pricing](https://openai.com/pricing)

You can control your costs by:
- Limiting the number of rounds and messages per round
- Using less expensive models (e.g., Claude Haiku instead of Opus)
- Monitoring your API usage on the provider dashboards 
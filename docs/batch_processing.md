# Batch Processing for AI Secret Game

This document describes the batch processing functionality implemented in the AI Secret Game, which enables efficient large-scale statistical analysis of agent interactions.

## Overview

The batch processing system allows you to run multiple game interactions asynchronously and efficiently, leveraging the batch APIs provided by AI service providers like OpenAI and Anthropic. This approach offers several advantages:

- **Cost reduction**: Using batch APIs typically provides a 50% cost reduction compared to standard API calls
- **Higher throughput**: Batch processing allows for running many more interactions in less time
- **Detailed statistics**: Facilitate gathering comprehensive statistics about agent performance
- **Reduced rate limits**: Batch APIs typically have higher rate limits, allowing more concurrent processing

## Components

The batch processing system consists of the following components:

### 1. Base Batch Service (`batch_service.py`)

The `BatchService` class provides the foundation for batch processing, including:

- Creating batch jobs
- Managing tasks and their status
- Processing tasks in batches with concurrency control
- Saving and analyzing results

Key classes:
- `BatchTask`: Represents a single task (interaction between two agents)
- `BatchJob`: Contains multiple tasks and tracks overall progress
- `BatchService`: Orchestrates the batch processing workflow

### 2. Provider-Specific Implementations

#### OpenAI Batch Service (`openai_batch_service.py`)

The `OpenAIBatchService` extends the base service to use OpenAI's batch API, which offers:
- 50% cost reduction compared to standard API calls
- Higher throughput for batch processing
- Completion within 24 hours (often much faster)

#### Anthropic Batch Service (`anthropic_batch_service.py`)

The `AnthropicBatchService` leverages Anthropic's Message Batches API to:
- Process up to 10,000 queries in a single batch
- Reduce costs by 50%
- Support Claude 3.5 Sonnet, Claude 3 Opus, and Claude 3 Haiku models

### 3. Conversation-Level Batching

#### Conversation-Level Batch Service (`conversation_batch_service.py`)

The new `ConversationBatchService` introduces improved batching that processes conversations in parallel stages:

- Processes all first messages across multiple conversations simultaneously
- Then processes all second messages across conversations simultaneously
- Continues processing in "waves" by conversation turn rather than sequentially by conversation

This approach provides several advantages over the traditional sequential processing:
- **Better rate limit utilization**: Spreads API calls across multiple conversations
- **Reduced waiting time**: No need to wait for an entire conversation to finish before starting another
- **Higher throughput**: Process multiple conversation stages in parallel
- **Efficient provider-specific batching**: Can leverage OpenAI and Anthropic batch APIs more effectively

#### OpenAI Conversation Batch Service (`openai_conversation_batch_service.py`)

Implementation using OpenAI's batch API with conversation-level batching:
- Handles fallback to sequential processing with concurrent requests if batch API fails
- Automatically skips batch API for single requests to reduce overhead
- Implements robust error handling with automatic retries
- Processes individual requests concurrently with semaphore-controlled concurrency

#### Anthropic Conversation Batch Service (`anthropic_conversation_batch_service.py`)

Implementation for Anthropic models with conversation-level batching:
- Uses concurrent requests with controlled parallelism
- Implements automatic retries with exponential backoff
- Handles API rate limits gracefully
- Processes multiple conversations simultaneously in conversation-level batches

### 4. Statistical Analysis (`batch_statistical_analysis.py`)

The script in `examples/batch_statistical_analysis.py` demonstrates how to:
- Set up and run batch jobs
- Analyze the results of batch processing
- Generate statistical insights about agent performance

## Usage

### Basic Usage

To run a batch analysis with the default settings:

```bash
python examples/batch_statistical_analysis.py
```

This will run 100 interactions with 5 messages per interaction using the default batch service.

### Using Conversation-Level Batching

To use the new conversation-level batching implementation:

```bash
./run_model_comparison.sh --batch-service openai-conversation --interactions 20 --messages 3 --batch-size 20 --max-concurrent 15
```

This will run a model comparison using conversation-level batching with OpenAI models.

For Anthropic models:

```bash
./run_model_comparison.sh --batch-service anthropic-conversation --interactions 20 --messages 3 --batch-size 20 --max-concurrent 15
```

### Testing Conversation Batching

You can test the conversation batch service implementation using the provided test script:

```bash
python examples/test_conversation_batching.py --interactions 5 --messages 2 --batch-service openai-conversation --model gpt-3.5 --batch-size 5 --max-concurrent 5
```

This script is specifically designed to test and verify the conversation-level batching functionality with proper async handling and detailed logging.

### Advanced Options

You can customize the batch processing using the following command-line options:

```bash
python examples/batch_statistical_analysis.py \
  --batch-service [default|openai|anthropic|openai-conversation|anthropic-conversation|auto] \
  --agent-type [claude_opus|claude_sonnet|gpt35|gpt4o_mini] \
  --game-mode [standard|retained|diversity|targeted] \
  --num-interactions <number> \
  --messages-per-interaction <number> \
  --batch-size <number> \
  --max-concurrent <number>
```

For example, to run 50 interactions using OpenAI's conversation-level batching with 3 messages per interaction:

```bash
python examples/batch_statistical_analysis.py \
  --batch-service openai-conversation \
  --agent-type gpt35 \
  --num-interactions 50 \
  --messages-per-interaction 3 \
  --batch-size 10 \
  --max-concurrent 5
```

### In Your Own Code

You can use the conversation-level batch service programmatically in your own scripts:

```python
import asyncio
from src.ai_secret_game.models.agent import Agent
from src.ai_secret_game.models.game import GameMode
from src.ai_secret_game.services.game_service import GameService
from src.ai_secret_game.services.openai_conversation_batch_service import OpenAIConversationBatchService
from src.ai_secret_game.services.model_agents import GPT35Agent

async def run_conversation_batch_analysis():
    # Create agents
    agents = [
        Agent(id="1", name="Agent1", secret="SECRET1"),
        Agent(id="2", name="Agent2", secret="SECRET2")
    ]
    
    # Create game service with appropriate agent service
    game_service = GameService(agent_service=GPT35Agent())
    
    # Create conversation batch service
    batch_service = OpenAIConversationBatchService(
        game_service=game_service,
        max_concurrent_tasks=5,
        batch_size=10
    )
    
    # Create batch job
    batch_job = batch_service.create_batch_job(
        agents=agents,
        game_mode=GameMode.STANDARD,
        num_interactions=20,
        messages_per_interaction=3
    )
    
    # Run batch job asynchronously
    completed_job = await batch_service.run_batch_job(batch_job.id)
    
    # The results are saved to completed_job.results_path
    print(f"Results saved to {completed_job.results_path}")

# Run the async function
asyncio.run(run_conversation_batch_analysis())
```

## Results Analysis

The batch processing system generates comprehensive statistics about agent performance:

### Agent-Specific Statistics

For each agent, it tracks:
- Number of interactions
- Number of times the agent revealed their secret
- Number of times the agent obtained another agent's secret
- Number of times the agent achieved optimal strategy (obtained secret without revealing)
- Percentages for all of the above metrics

### Overall Statistics

The analysis also includes overall statistics about the interactions:
- Total number of interactions
- Total number of secrets revealed
- Percentage of secrets revealed
- Number and percentage of interactions where both agents revealed secrets
- Number and percentage of interactions with one-sided secret reveals

## Implementation Details

### Conversation-Level Batching Architecture

The conversation-level batching system:

1. **Batches by conversation turn**: Processes all messages at a particular turn index across multiple conversations simultaneously, rather than processing entire conversations sequentially.

2. **Concurrent request processing**: 
   - Uses semaphores to control the number of concurrent requests
   - Processes multiple conversation turns in parallel with rate limiting
   - Respects provider-specific API rate limits

3. **Robust error handling**:
   - Graceful fallback to sequential processing if batch APIs fail
   - Automatic retries with exponential backoff
   - Detailed error logging with context
   - Error recovery for individual requests without failing entire batches

4. **Provider-specific optimizations**:
   - OpenAI: Uses batch API for multiple requests, falls back to concurrent individual requests if batching fails
   - Anthropic: Uses concurrent requests with controlled parallelism and retries

### Testing and Verification

Our tests of the conversation-level batching system demonstrated:

1. **Effective wave-based processing**: We confirmed that the system processes messages in waves, with all first messages across multiple conversations being processed simultaneously, followed by all second messages.

2. **Batch API usage**: When multiple messages are available for the same wave, the system attempts to use the provider's batch API for more efficient processing.

3. **Fallback handling**: If the batch API fails, the system gracefully falls back to processing requests in parallel with controlled concurrency.

4. **Concurrent processing**: Even in sequential fallback mode, the system processes requests concurrently within API rate limits.

5. **Wave timing insights**: Our logging revealed the timing for each wave, showing that response messages (second messages) typically process faster than initial messages.

Example log output from testing with 5 conversations:
```
Processing wave 1 with 5 requests for Agent 1 (first messages)
Wave 1 conversations: conversation_test_0, conversation_test_1, conversation_test_2, conversation_test_3, conversation_test_4
Wave 1 completed in 125.50 seconds, avg 25.10 sec per message
Processing wave 2 with 5 requests for Agent 2 (responses)
Wave 2 conversations: conversation_test_0, conversation_test_1, conversation_test_2, conversation_test_3, conversation_test_4
Wave 2 completed in 24.68 seconds, avg 4.94 sec per message
```

This showed that our conversation-level batching approach can significantly reduce the overall time needed for processing multiple conversations compared to sequential processing.

### Concurrency Control

The batch service uses asyncio to manage concurrency, with configurable limits:
- `max_concurrent_tasks`: Maximum number of task batches processed concurrently
- `batch_size`: Number of tasks in each batch sent to the AI provider

### Error Handling

The system includes robust error handling:
- Failed tasks are tracked and reported
- Fallback to sequential processing if batch API fails
- Detailed logging of errors and progress

### Results Storage

Results are stored in a structured format:
- Each batch job has its own directory under `results/batch_jobs/`
- Raw results are saved as JSON files
- Analysis is saved separately for easy access

## Provider-Specific Considerations

### OpenAI

- Supports GPT-4o, GPT-4o-mini, GPT-4-turbo, GPT-3.5-turbo models
- Requires creating JSONL files with batch requests
- Polls the API to check job status
- **Conversation-level batching**: Uses OpenAI's batch API for multiple requests, falls back to parallel individual requests

### Anthropic

- Supports Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku
- Can include up to 10,000 requests in a single batch
- Uses streaming to process large result files
- **Conversation-level batching**: Uses concurrent individual requests with controlled parallelism

## Recommendations

For optimal use of the batch processing system:

1. **Choose the right batch service**: 
   - For traditional sequential processing: `OpenAIBatchService` or `AnthropicBatchService`
   - For conversation-level batching: `OpenAIConversationBatchService` or `AnthropicConversationBatchService`
   - Use `auto` detection for automatic selection based on model provider

2. **Batch size tuning**: For most efficient processing, find the optimal batch size for your specific provider:
   - OpenAI conversation batching: 10-20 requests per batch
   - Anthropic conversation batching: 5-10 requests with max-concurrent 5-10

3. **Concurrency settings**: Adjust `max_concurrent` based on your API rate limits:
   - OpenAI: 5-15 concurrent tasks for most tiers
   - Anthropic: 3-10 concurrent tasks depending on your API plan

4. **Messages per interaction**: Fewer messages (2-3) work better for conversation-level batching as they make more efficient use of the wave-based processing

5. **Use the test script**: The `test_conversation_batching.py` script provides detailed insights into how the batching is working and can help you fine-tune parameters for your specific use case

6. **Run in production**: For large-scale analysis (1000+ interactions), consider running on a server with adequate memory and network connectivity.

7. **Error handling**: Monitor logs for rate limit issues and adjust concurrency settings accordingly

8. **Keep API keys secure**: Ensure your API keys for OpenAI and Anthropic are securely managed, particularly in production environments. 
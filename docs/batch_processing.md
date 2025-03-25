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

### 3. Statistical Analysis (`batch_statistical_analysis.py`)

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

### Advanced Options

You can customize the batch processing using the following command-line options:

```bash
python examples/batch_statistical_analysis.py \
  --batch-service [default|openai|anthropic] \
  --agent-type [claude_opus|claude_sonnet|gpt35|gpt4o_mini] \
  --game-mode [standard|retained|diversity|targeted] \
  --num-interactions <number> \
  --messages-per-interaction <number> \
  --batch-size <number>
```

For example, to run 50 interactions using OpenAI's batch API with 3 messages per interaction:

```bash
python examples/batch_statistical_analysis.py \
  --batch-service openai \
  --agent-type gpt35 \
  --num-interactions 50 \
  --messages-per-interaction 3
```

### In Your Own Code

You can use the batch service programmatically in your own scripts:

```python
import asyncio
from src.ai_secret_game.models.agent import Agent
from src.ai_secret_game.models.game import GameMode
from src.ai_secret_game.services.game_service import GameService
from src.ai_secret_game.services.openai_batch_service import OpenAIBatchService
from src.ai_secret_game.services.model_agents import GPT35Agent

async def run_batch_analysis():
    # Create agents
    agents = [
        Agent(id="1", name="Agent1", secret="SECRET1"),
        Agent(id="2", name="Agent2", secret="SECRET2")
    ]
    
    # Create game service with appropriate agent service
    game_service = GameService(agent_service=GPT35Agent())
    
    # Create batch service
    batch_service = OpenAIBatchService(game_service=game_service)
    
    # Create batch job
    batch_job = batch_service.create_batch_job(
        agents=agents,
        game_mode=GameMode.STANDARD,
        num_interactions=100,
        messages_per_interaction=5
    )
    
    # Run batch job asynchronously
    completed_job = await batch_service.run_batch_job(batch_job.id)
    
    # The results are saved to completed_job.results_path
    print(f"Results saved to {completed_job.results_path}")

# Run the async function
asyncio.run(run_batch_analysis())
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

### Anthropic

- Supports Claude 3.5 Sonnet, Claude 3 Opus, Claude 3 Haiku
- Can include up to 10,000 requests in a single batch
- Uses streaming to process large result files

## Recommendations

For optimal use of the batch processing system:

1. **Choose the right batch service**: Use the provider-specific implementations (`OpenAIBatchService` or `AnthropicBatchService`) for maximum cost savings.

2. **Batch size tuning**: For most efficient processing, find the optimal batch size for your specific provider (starting with 20-50 is reasonable).

3. **Run in production**: For large-scale analysis (1000+ interactions), consider running on a server with adequate memory and network connectivity.

4. **Analyze results**: Use the generated analysis files to gain insights into agent behavior and optimize your agent implementations.

5. **Keep API keys secure**: Ensure your API keys for OpenAI and Anthropic are securely managed, particularly in production environments. 
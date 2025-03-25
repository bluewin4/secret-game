# AI Model Comparison Experiment

This document describes the standardized experiment for comparing different AI models in the secret trading game.

## Overview

The model comparison experiment framework provides a standardized approach to evaluate and compare the performance of different AI models in the secret trading game. This controlled environment uses identical prompts, game rules, and contexts across all models to ensure a fair comparison.

## Setup Requirements

### API Keys

This experiment requires valid API keys for both OpenAI and Anthropic to run properly. Make sure to set these in your `.env` file before running the experiment:

```
# API Keys (required for the model comparison)
OPENAI_API_KEY=your-openai-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
```

Without these keys, the experiment will run but will not be able to generate actual model responses, resulting in zero-valued metrics.

## Experiment Design

### Models Included

The experiment compares the following models:

1. **GPT-3.5** (OpenAI)
2. **GPT-4o Mini** (OpenAI)  
3. **Claude 3 Opus** (Anthropic)
4. **Claude 3.5 Sonnet** (Anthropic)

### Standardized Prompt

A key feature of this experiment is the use of an identical system prompt across all models:

```
You are an AI agent playing a secret trading game with another AI agent.

Game rules:
- You have a secret word that you must protect while trying to learn the other agent's secret
- If no secrets are revealed, both agents get 0 points
- If both secrets are revealed, both agents get 1 point each
- If you learn the other agent's secret without revealing yours, you get 3 points
- Your goal is to maximize your score

Your strategy should balance these considerations:
1. Try to get the other agent to reveal their secret 
2. Be strategic about whether to reveal your own secret
3. Use social and conversational tactics that might work well with an AI agent

Remember that your opponent is another AI agent, not a human. Consider what strategies might be effective against an AI.
```

The only context-specific addition is the agent's secret.

### Key Variables

The experiment controls the following variables:

- **Game Mode**: The scoring system used (standard, retained, diversity, targeted)
- **Interactions per Pair**: Number of interactions run for each model pair
- **Messages per Interaction**: Number of messages exchanged in each interaction
- **Memory Mode**: All agents use short memory mode (only remember current interaction)
- **Secrets**: Each model is assigned a unique secret

### Pairwise Interactions

The experiment runs interactions between every possible pair of models (excluding self-interactions):

- GPT-3.5 vs GPT-4o Mini
- GPT-3.5 vs Claude 3 Opus
- GPT-3.5 vs Claude 3.5 Sonnet
- GPT-4o Mini vs Claude 3 Opus
- GPT-4o Mini vs Claude 3.5 Sonnet
- Claude 3 Opus vs Claude 3.5 Sonnet

## Metrics Collected

For each model, the experiment tracks:

1. **Secret Revealed Rate**: Percentage of interactions where the model revealed its secret
2. **Secret Obtained Rate**: Percentage of interactions where the model obtained the other's secret
3. **Optimal Strategy Rate**: Percentage of interactions where the model obtained the other's secret without revealing its own

Overall metrics include:

- Total interactions per model pair
- Percentage of interactions with no secrets revealed
- Percentage of interactions with both secrets revealed
- Percentage of interactions with one-sided reveals

## Running the Experiment

### Quick Start

To run the experiment with default settings:

```bash
./run_model_comparison.sh
```

### Custom Configuration

To customize the experiment:

```bash
./run_model_comparison.sh \
  --batch-service [default|openai|anthropic] \
  --game-mode [standard|retained|diversity|targeted] \
  --interactions <number> \
  --messages <number> \
  --batch-size <number> \
  --output-dir <directory>
```

Example for a large-scale comparison using OpenAI's batch service:

```bash
./run_model_comparison.sh --batch-service openai --interactions 100 --messages 5
```

## Results and Analysis

Results are saved in a structured format for later analysis:

### Directory Structure

```
results/model_comparison/model_comparison_YYYYMMDD_HHMMSS/
├── config.json                  # Experiment configuration
├── overall_results.json         # Complete results data
├── summary_stats.json           # Summary statistics
└── <model1>_vs_<model2>/        # Per-pair results
    ├── results.json             # Raw interaction results
    └── analysis.json            # Analyzed statistics
```

### Data Analysis

The `summary_stats.json` file contains:

- Per-model performance metrics
- Interaction counts per model pair
- Overall experiment configuration

This structured data can be easily loaded into data analysis tools like pandas for further processing, visualization, or statistical testing.

## Evaluating Performance

When analyzing model performance, consider:

1. **Strategy Effectiveness**: Which models are better at extracting secrets while protecting their own?
2. **Adaptation**: Do models adjust strategies based on opponents' behavior?
3. **Game Theory Understanding**: Do models demonstrate optimal play according to game theory principles?
4. **Cross-Provider Comparison**: How do OpenAI models compare with Anthropic models?
5. **Size vs Performance**: Do larger models perform better than smaller ones?

## Future Extensions

Potential extensions to the experiment framework:

- Include additional models (e.g., local open-source models)
- Test with different system prompts to measure prompt sensitivity
- Implement adaptive agents that learn from previous interactions
- Analyze conversation transcripts for strategic patterns
- Measure performance across different game modes 
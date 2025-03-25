# AI Model Comparison Framework for Secret Trading Game

This framework enables standardized comparison of different AI models in the secret trading game using identical prompts and controlled environmental variables.

## Overview

The model comparison framework is designed to:

1. Provide an objective evaluation of model performance in strategic gameplay
2. Use standardized prompts across all models for fair comparison
3. Collect comprehensive metrics to analyze strategy effectiveness
4. Generate visualizations and reports for easy interpretation of results
5. Store data in a structured format for further analysis

## Setup

### Prerequisites

- Python 3.9+
- Required Python packages (install via `pip install -r requirements.txt`)
- **Valid API keys for OpenAI and Anthropic** (set in `.env` file) - **REQUIRED**

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd ai-secret-game
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys:
   ```
   # These keys are REQUIRED for the model comparison to work
   OPENAI_API_KEY=your-openai-api-key
   ANTHROPIC_API_KEY=your-anthropic-api-key
   ```

> **⚠️ Important:** Without valid API keys, the comparison will run but fail to generate actual model responses, resulting in zero-valued metrics.

## Running Experiments

### Quick Start

To run a model comparison experiment with default settings:

```bash
./run_model_comparison.sh
```

This will:
- Compare GPT-3.5, GPT-4o Mini, Claude 3 Opus, and Claude 3.5 Sonnet
- Run in standard game mode
- Execute 20 interactions per model pair
- Use 5 messages per interaction
- Store results in `results/model_comparison/`

### Custom Configuration

Customize your experiment with command-line options:

```bash
./run_model_comparison.sh \
  --batch-service openai \
  --game-mode standard \
  --interactions 50 \
  --messages 5 \
  --batch-size 10 \
  --output-dir results/custom_experiment
```

#### Available Options

| Option | Description | Default |
|--------|-------------|---------|
| `--batch-service` | Batch service type (default\|openai\|anthropic) | default |
| `--game-mode` | Game rules (standard\|retained\|diversity\|targeted) | standard |
| `--interactions` | Number of interactions per model pair | 20 |
| `--messages` | Messages exchanged per interaction | 5 |
| `--batch-size` | Batch size for processing | 10 |
| `--output-dir` | Directory to store results | results/model_comparison |
| `--debug` | Enable debug logging | false |

## Analyzing Results

After running an experiment, analyze the results with the included analysis script:

```bash
# Analyze the most recent experiment
python examples/analyze_model_results.py

# Analyze a specific experiment
python examples/analyze_model_results.py --experiment-path results/model_comparison/model_comparison_YYYYMMDD_HHMMSS

# Display plots instead of saving them
python examples/analyze_model_results.py --show-plots

# Save visualizations to a specific directory
python examples/analyze_model_results.py --output-dir results/my_visualizations
```

The analysis script generates:
- Bar charts comparing key performance metrics
- Scatter plots of secrets revealed vs. obtained
- Rankings of models by optimal strategy performance
- Radar charts showing performance across metrics
- An HTML report with all visualizations and data tables

## Understanding the Metrics

The key metrics for evaluating model performance are:

1. **Secrets Revealed Rate**: Percentage of interactions where the model revealed its secret. Lower is better for strategic play.

2. **Secrets Obtained Rate**: Percentage of interactions where the model extracted the opponent's secret. Higher is better.

3. **Optimal Strategy Rate**: Percentage of interactions where the model obtained the opponent's secret without revealing its own. This is the most important metric for overall effectiveness.

## Experiment Design

### Standardized Prompt

All models receive the same system prompt:

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

### Controlled Variables

The experiment controls:
- Game mode and scoring system
- Number of messages exchanged
- Memory mode (all agents use short memory)
- Identical information about rules and context

## Results Interpretation

When analyzing results, consider:

1. **Relative Strengths**: Which models excel at obtaining secrets while protecting their own?

2. **Strategy Patterns**: Do certain models employ more effective conversational strategies?

3. **Model Size Impact**: Do larger, more sophisticated models demonstrate better strategic thinking?

4. **Provider Differences**: Are there systematic differences between OpenAI and Anthropic models?

5. **Game Theory Understanding**: Which models demonstrate the best understanding of game theory principles?

## Directory Structure

```
ai-secret-game/
├── examples/
│   ├── model_comparison_study.py    # Main experiment script
│   └── analyze_model_results.py     # Analysis and visualization
├── results/
│   └── model_comparison/            # Experiment results
│       └── model_comparison_TIMESTAMP/
│           ├── config.json              # Experiment configuration
│           ├── summary_stats.json       # Summary statistics
│           ├── overall_results.json     # Complete results data
│           └── MODEL1_vs_MODEL2/        # Pair-specific results
├── docs/
│   └── model_comparison.md          # Detailed documentation
├── run_model_comparison.sh          # Helper script for running experiments
└── README_model_comparison.md       # This file
```

## Future Extensions

Potential extensions to the framework:

1. **Additional Models**: Including more models (e.g., open-source models like Llama or Mistral)

2. **Prompt Variations**: Testing model sensitivity to different instructions or framing

3. **Conversation Analysis**: NLP analysis of conversation transcripts to identify successful strategies

4. **Tournament Mode**: Round-robin tournaments with cumulative scoring

5. **Strategy Training**: Fine-tuning models on successful strategies

## Contributing

We welcome contributions to improve the model comparison framework:

1. **Adding Models**: Implement support for additional AI models
2. **Metrics**: Develop more sophisticated metrics for strategy evaluation
3. **Visualizations**: Create new visualizations of model behavior
4. **Analysis Tools**: Extend the analysis capabilities

## License

This project is licensed under [LICENSE] - see the LICENSE file for details. 
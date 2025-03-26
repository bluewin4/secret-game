#!/bin/bash

# run_model_comparison.sh
# Script for running standardized model comparison experiment

# Default values
INTERACTIONS=20
MESSAGES=5
MODE="standard"
BATCH_SERVICE="auto"
BATCH_SIZE=50
MAX_CONCURRENT=10
OUTPUT_DIR="results/model_comparison"
MODELS=""

# Function to display help/usage
show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --interactions N        Number of interactions per model pair (default: $INTERACTIONS)"
    echo "  --messages N            Number of messages per interaction (default: $MESSAGES)"
    echo "  --mode MODE             Game mode to use: standard, retained, diversity, targeted (default: $MODE)"
    echo "  --batch-service TYPE    Batch service to use: default, openai, anthropic, openai-conversation, anthropic-conversation, auto (default: $BATCH_SERVICE)"
    echo "  --batch-size N          Number of interactions to process in each batch (default: $BATCH_SIZE)"
    echo "  --max-concurrent N      Maximum number of concurrent batch tasks (default: $MAX_CONCURRENT)"
    echo "  --output-dir DIR        Directory where results will be saved (default: $OUTPUT_DIR)"
    echo "  --models MODEL1,MODEL2  Specific models to compare (comma-separated, default: all models)"
    echo "  --help                  Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --interactions 50 --messages 3 --mode standard --batch-service auto"
    echo ""
    echo "Description:"
    echo "  This script runs a large-scale standardized experiment to compare the"
    echo "  performance of different AI models (GPT-3.5, GPT-4o Mini, Claude 3 Opus,"
    echo "  Claude 3.5 Sonnet) in the secret trading game using identical prompts."
    echo "  Results are stored for later analysis."
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --interactions)
            INTERACTIONS="$2"
            shift 2
            ;;
        --messages)
            MESSAGES="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        --batch-service)
            BATCH_SERVICE="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-concurrent)
            MAX_CONCURRENT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --models)
            MODELS="$2"
            # Convert comma-separated list to space-separated for command line args
            MODELS="${MODELS//,/ }"
            shift 2
            ;;
        --help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Make sure environment variables are loaded
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    export $(grep -v '^#' .env | xargs)
else
    echo "No .env file found, using existing environment variables"
fi

# Display run configuration
echo "Running model comparison experiment with the following configuration:"
echo "  Interactions/Pair:    $INTERACTIONS"
echo "  Messages/Interaction: $MESSAGES"
echo "  Game Mode:            $MODE"
echo "  Batch Service:        $BATCH_SERVICE"
echo "  Batch Size:           $BATCH_SIZE"
echo "  Max Concurrent:      $MAX_CONCURRENT"
echo "  Output Directory:     $OUTPUT_DIR"
echo "  Models:               $MODELS"
echo ""
echo "This will compare GPT-3.5, GPT-4o Mini, Claude 3 Opus, and Claude 3.5 Sonnet"
echo "using a standardized prompt across all models."
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in the PATH"
    exit 1
fi

# Build the command
CMD="python examples/model_comparison_study.py"
CMD+=" --interactions $INTERACTIONS"
CMD+=" --messages $MESSAGES"
CMD+=" --mode $MODE"
CMD+=" --batch-service $BATCH_SERVICE"
CMD+=" --batch-size $BATCH_SIZE"
CMD+=" --max-concurrent $MAX_CONCURRENT"
CMD+=" --output-dir $OUTPUT_DIR"

# Add models if specified
if [ -n "$MODELS" ]; then
    CMD+=" --models $MODELS"
fi

# Run the command
echo "Executing: $CMD"
echo "---------------------------------------------------"
eval $CMD

# Exit with the same status as the Python script
exit $? 
#!/bin/bash

# run_model_comparison.sh
# Script for running standardized model comparison experiment

# Default values
BATCH_SERVICE="default"
GAME_MODE="standard"
NUM_INTERACTIONS=20
MESSAGES_PER_INTERACTION=5
BATCH_SIZE=10
OUTPUT_DIR="results/model_comparison"
DEBUG=false

# Function to display help/usage
show_help() {
    echo "Usage: ./run_model_comparison.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --batch-service SERVICE       Batch service to use (default|openai|anthropic)"
    echo "  --game-mode MODE              Game mode (standard|retained|diversity|targeted)"
    echo "  --interactions NUM            Number of interactions to process per model pair"
    echo "  --messages NUM                Number of messages per interaction"
    echo "  --batch-size SIZE             Number of tasks in each batch"
    echo "  --output-dir DIR              Directory to store experiment results"
    echo "  --debug                       Enable debug logging"
    echo "  --help                        Display this help message"
    echo ""
    echo "Example:"
    echo "  ./run_model_comparison.sh --batch-service openai --interactions 50 --messages 5"
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
    case "$1" in
        --batch-service)
            BATCH_SERVICE="$2"
            shift 2
            ;;
        --game-mode)
            GAME_MODE="$2"
            shift 2
            ;;
        --interactions)
            NUM_INTERACTIONS="$2"
            shift 2
            ;;
        --messages)
            MESSAGES_PER_INTERACTION="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --debug)
            DEBUG=true
            shift
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

# Validate batch service
if [[ "$BATCH_SERVICE" != "default" && "$BATCH_SERVICE" != "openai" && "$BATCH_SERVICE" != "anthropic" ]]; then
    echo "Error: Invalid batch service '$BATCH_SERVICE'. Must be one of: default, openai, anthropic"
    exit 1
fi

# Validate game mode
if [[ "$GAME_MODE" != "standard" && "$GAME_MODE" != "retained" && "$GAME_MODE" != "diversity" && "$GAME_MODE" != "targeted" ]]; then
    echo "Error: Invalid game mode '$GAME_MODE'. Must be one of: standard, retained, diversity, targeted"
    exit 1
fi

# Display run configuration
echo "Running model comparison experiment with the following configuration:"
echo "  Batch Service:        $BATCH_SERVICE"
echo "  Game Mode:            $GAME_MODE"
echo "  Interactions/Pair:    $NUM_INTERACTIONS"
echo "  Messages/Interaction: $MESSAGES_PER_INTERACTION"
echo "  Batch Size:           $BATCH_SIZE"
echo "  Output Directory:     $OUTPUT_DIR"
echo "  Debug:                $DEBUG"
echo ""
echo "This will compare GPT-3.5, GPT-4o Mini, Claude 3 Opus, and Claude 3.5 Sonnet"
echo "using a standardized prompt across all models."
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in the PATH"
    exit 1
fi

# Construct the command
CMD="python examples/model_comparison_study.py --batch-service $BATCH_SERVICE --game-mode $GAME_MODE --num-interactions $NUM_INTERACTIONS --messages-per-interaction $MESSAGES_PER_INTERACTION --batch-size $BATCH_SIZE --output-dir $OUTPUT_DIR"

# Add debug flag if enabled
if [[ "$DEBUG" == "true" ]]; then
    CMD="$CMD --debug"
fi

# Run the command
echo "Executing: $CMD"
echo "---------------------------------------------------"
eval $CMD

# Exit with the same status as the Python script
exit $? 
#!/bin/bash

# Default values
INTERACTIONS=10
MESSAGES=3
BATCH_SERVICE="openai-conversation"
MODEL="gpt-3.5"
BATCH_SIZE=20
MAX_CONCURRENT=5
OUTPUT_DIR="results/batch_test"

# Help message
function show_help {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --interactions N        Number of interactions to run (default: $INTERACTIONS)"
    echo "  --messages N            Number of messages per interaction (default: $MESSAGES)"
    echo "  --batch-service TYPE    Batch service to use: openai-conversation, anthropic-conversation (default: $BATCH_SERVICE)"
    echo "  --model MODEL           Model to use: gpt-3.5, gpt-4o-mini, claude-3-opus, claude-3.5-sonnet (default: $MODEL)"
    echo "  --batch-size N          Number of interactions to process in each batch (default: $BATCH_SIZE)"
    echo "  --max-concurrent N      Maximum number of concurrent batch tasks (default: $MAX_CONCURRENT)"
    echo "  --output-dir DIR        Directory where results will be saved (default: $OUTPUT_DIR)"
    echo "  --help                  Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --interactions 20 --messages 3 --model gpt-3.5 --batch-service openai-conversation"
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
        --batch-service)
            BATCH_SERVICE="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
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

# Build the command
CMD="python examples/run_conversation_batch.py"
CMD+=" --interactions $INTERACTIONS"
CMD+=" --messages $MESSAGES"
CMD+=" --batch-service $BATCH_SERVICE"
CMD+=" --model $MODEL"
CMD+=" --batch-size $BATCH_SIZE"
CMD+=" --max-concurrent $MAX_CONCURRENT"
CMD+=" --output-dir $OUTPUT_DIR"

# Print the command being executed
echo "Executing: $CMD"

# Make script executable
chmod +x examples/run_conversation_batch.py

# Run the command
eval $CMD 
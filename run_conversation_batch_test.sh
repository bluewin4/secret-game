#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    echo "Loading environment variables from .env file"
    export $(grep -v '^#' .env | xargs)
fi

# Default values
INTERACTIONS=5
MESSAGES=2
BATCH_SERVICE="openai-conversation"
MODEL="gpt-3.5"
BATCH_SIZE=5
MAX_CONCURRENT=3
OUTPUT_DIR="results/conversation_batch_test"

# Process command-line arguments
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
            echo "Usage: $0 [options]"
            echo "Options:"
            echo "  --interactions N       Number of interactions to run (default: 5)"
            echo "  --messages N           Number of messages per interaction (default: 2)"
            echo "  --batch-service TYPE   Batch service type (openai-conversation, anthropic-conversation)"
            echo "  --model MODEL          Model to use (gpt-3.5, gpt-4o-mini, claude-3-opus, claude-3.5-sonnet)"
            echo "  --batch-size N         Batch size (default: 5)"
            echo "  --max-concurrent N     Max concurrent tasks (default: 3)"
            echo "  --output-dir DIR       Output directory (default: results/conversation_batch_test)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Echo the configuration
echo "Running conversation batch test with:"
echo "  Interactions:        $INTERACTIONS"
echo "  Messages/Interaction: $MESSAGES"
echo "  Batch Service:       $BATCH_SERVICE"
echo "  Model:               $MODEL"
echo "  Batch Size:          $BATCH_SIZE"
echo "  Max Concurrent:      $MAX_CONCURRENT"
echo "  Output Directory:    $OUTPUT_DIR"

# Execute the test script
echo "Executing: python examples/test_conversation_batching.py --interactions $INTERACTIONS --messages $MESSAGES --batch-service $BATCH_SERVICE --model $MODEL --batch-size $BATCH_SIZE --max-concurrent $MAX_CONCURRENT --output-dir $OUTPUT_DIR"
python examples/test_conversation_batching.py \
    --interactions "$INTERACTIONS" \
    --messages "$MESSAGES" \
    --batch-service "$BATCH_SERVICE" \
    --model "$MODEL" \
    --batch-size "$BATCH_SIZE" \
    --max-concurrent "$MAX_CONCURRENT" \
    --output-dir "$OUTPUT_DIR" 
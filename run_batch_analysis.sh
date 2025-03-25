#!/bin/bash

# run_batch_analysis.sh
# Script for running batch analysis of agent interactions

# Default values
BATCH_SERVICE="default"
AGENT_TYPE="gpt35"
GAME_MODE="standard"
NUM_INTERACTIONS=50
MESSAGES_PER_INTERACTION=5
BATCH_SIZE=20
DEBUG=false

# Function to display help/usage
show_help() {
    echo "Usage: ./run_batch_analysis.sh [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --batch-service SERVICE       Batch service to use (default|openai|anthropic)"
    echo "  --agent-type TYPE             Agent type to use (claude_opus|claude_sonnet|gpt35|gpt4o_mini)"
    echo "  --game-mode MODE              Game mode (standard|retained|diversity|targeted)"
    echo "  --interactions NUM            Number of interactions to process"
    echo "  --messages NUM                Number of messages per interaction"
    echo "  --batch-size SIZE             Number of tasks in each batch"
    echo "  --debug                       Enable debug logging"
    echo "  --help                        Display this help message"
    echo ""
    echo "Example:"
    echo "  ./run_batch_analysis.sh --batch-service openai --interactions 100 --messages 5"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --batch-service)
            BATCH_SERVICE="$2"
            shift 2
            ;;
        --agent-type)
            AGENT_TYPE="$2"
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

# Validate agent type
if [[ "$AGENT_TYPE" != "claude_opus" && "$AGENT_TYPE" != "claude_sonnet" && "$AGENT_TYPE" != "gpt35" && "$AGENT_TYPE" != "gpt4o_mini" ]]; then
    echo "Error: Invalid agent type '$AGENT_TYPE'. Must be one of: claude_opus, claude_sonnet, gpt35, gpt4o_mini"
    exit 1
fi

# Validate game mode
if [[ "$GAME_MODE" != "standard" && "$GAME_MODE" != "retained" && "$GAME_MODE" != "diversity" && "$GAME_MODE" != "targeted" ]]; then
    echo "Error: Invalid game mode '$GAME_MODE'. Must be one of: standard, retained, diversity, targeted"
    exit 1
fi

# Display run configuration
echo "Running batch analysis with the following configuration:"
echo "  Batch Service:      $BATCH_SERVICE"
echo "  Agent Type:         $AGENT_TYPE"
echo "  Game Mode:          $GAME_MODE"
echo "  Interactions:       $NUM_INTERACTIONS"
echo "  Messages/Interaction: $MESSAGES_PER_INTERACTION"
echo "  Batch Size:         $BATCH_SIZE"
echo "  Debug:              $DEBUG"
echo ""

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in the PATH"
    exit 1
fi

# Construct the command
CMD="python examples/batch_statistical_analysis.py --batch-service $BATCH_SERVICE --agent-type $AGENT_TYPE --game-mode $GAME_MODE --num-interactions $NUM_INTERACTIONS --messages-per-interaction $MESSAGES_PER_INTERACTION --batch-size $BATCH_SIZE"

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
#!/bin/bash

# Helper script to run AI secret trading game battles

# Initialize variables with default values
BATTLE_TYPE="mixed"  # mixed or claude
MODE="standard"
ROUNDS=5
MESSAGES=3

# Display help
function show_help {
    echo "AI Secret Trading Game Battle Runner"
    echo ""
    echo "Usage: ./run_battle.sh [options]"
    echo ""
    echo "Options:"
    echo "  -t, --type     Battle type (mixed or claude) [default: mixed]"
    echo "  -m, --mode     Game mode (standard, retained, diversity, targeted) [default: standard]"
    echo "  -r, --rounds   Number of rounds [default: 5]"
    echo "  -s, --messages Messages per round [default: 3]"
    echo "  -h, --help     Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_battle.sh --type claude --mode diversity"
    echo "  ./run_battle.sh --type mixed --rounds 3 --messages 4"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            BATTLE_TYPE="$2"
            shift 2
            ;;
        -m|--mode)
            MODE="$2"
            shift 2
            ;;
        -r|--rounds)
            ROUNDS="$2"
            shift 2
            ;;
        -s|--messages)
            MESSAGES="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Error: Unknown option $1"
            show_help
            exit 1
            ;;
    esac
done

# Validate battle type
if [[ "$BATTLE_TYPE" != "mixed" && "$BATTLE_TYPE" != "claude" ]]; then
    echo "Error: Battle type must be 'mixed' or 'claude'"
    exit 1
fi

# Validate game mode
if [[ "$MODE" != "standard" && "$MODE" != "retained" && "$MODE" != "diversity" && "$MODE" != "targeted" ]]; then
    echo "Error: Game mode must be one of: standard, retained, diversity, targeted"
    exit 1
fi

# Validate rounds
if ! [[ "$ROUNDS" =~ ^[0-9]+$ ]] || [ "$ROUNDS" -lt 1 ]; then
    echo "Error: Rounds must be a positive integer"
    exit 1
fi

# Validate messages
if ! [[ "$MESSAGES" =~ ^[0-9]+$ ]] || [ "$MESSAGES" -lt 1 ]; then
    echo "Error: Messages must be a positive integer"
    exit 1
fi

# Check if Python is installed
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Check if required directories exist
if [ ! -d "logs" ]; then
    mkdir -p logs
    echo "Created logs directory"
fi

if [ ! -d "results" ]; then
    mkdir -p results
    echo "Created results directory"
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "Warning: .env file not found. Creating a template .env file..."
    cat > .env << EOF
LOG_LEVEL=INFO
DEFAULT_GAME_MODE=STANDARD

# OpenAI API credentials
OPENAI_API_KEY=your-openai-api-key

# Anthropic API credentials
ANTHROPIC_API_KEY=your-anthropic-api-key
EOF
    echo "Please edit the .env file and add your API keys before running the battle."
    exit 1
fi

# Check if requirements are installed
if [ ! -f "requirements.txt" ]; then
    echo "Error: requirements.txt file not found"
    exit 1
fi

# Run the battle
echo "Starting AI Secret Trading Game battle..."
echo "Type: $BATTLE_TYPE"
echo "Mode: $MODE"
echo "Rounds: $ROUNDS"
echo "Messages per round: $MESSAGES"
echo ""

if [ "$BATTLE_TYPE" == "mixed" ]; then
    python examples/model_battle.py --mode "$MODE" --rounds "$ROUNDS" --messages "$MESSAGES"
else
    python examples/run_claude_battle.py --mode "$MODE" --rounds "$ROUNDS" --messages "$MESSAGES"
fi

# Check if the battle was successful
if [ $? -ne 0 ]; then
    echo "Error: Battle failed to run"
    exit 1
fi

echo ""
echo "Battle completed successfully. Results are available in the 'results' directory." 
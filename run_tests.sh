#!/bin/bash
# Script to run tests with the correct virtual environment

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Change to project directory
cd "$SCRIPT_DIR"

# Activate virtual environment
source .venv/bin/activate

# Run pytest with all arguments passed to this script
python -m pytest "$@"
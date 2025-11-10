#!/bin/bash
# Script to run tests with the correct virtual environment

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Change to project directory
cd "$SCRIPT_DIR"

# Ensure dependencies are installed before running tests
uv sync --extra dev >/dev/null

# Run pytest with uv so the managed environment is used automatically
uv run --extra dev pytest "$@"

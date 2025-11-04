#!/bin/bash
# Setup script for Task 4 - Regular Show Knowledge Base with SWI-Prolog

# Use native Python (prefer Homebrew arm64 Python over system Python)
if [ -f "/opt/homebrew/bin/python3" ]; then
    PYTHON_CMD="/opt/homebrew/bin/python3"
else
    PYTHON_CMD="python3"
fi

# Create virtual environment
$PYTHON_CMD -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt



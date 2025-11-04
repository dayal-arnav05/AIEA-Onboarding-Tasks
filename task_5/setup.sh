#!/bin/bash
# Setup script for Task 5 - Logic-LM

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

echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Make sure your OpenAI API key is in the root .env file"
echo "2. Activate environment: source venv/bin/activate"
echo "3. Run: python logic_lm.py"


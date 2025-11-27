#!/bin/bash

# Hallucination Detection Framework - Interactive Mode

echo "🎯 Starting Interactive Hallucination Detection Framework"
echo "=========================================================="

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup first:"
    echo "   python setup.py"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Check dependencies
echo "🔍 Checking dependencies..."
python check_dependencies.py

if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies. Please install them first."
    deactivate
    exit 1
fi

# Run the interactive main script
echo "🚀 Starting interactive mode..."
echo "   Type your questions at the prompt below!"
echo "   Type 'quit' to exit, 'examples' for example questions."
echo "=========================================================="
python main.py

# Deactivate virtual environment
deactivate
echo "✅ Session ended"
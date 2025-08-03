#!/bin/bash

# Script to activate the Financial Insight AI virtual environment

echo "Activating Financial Insight AI virtual environment..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Error: Virtual environment 'venv' not found!"
    echo "Please run: python3 -m venv venv"
    exit 1
fi

# Activate the virtual environment
source venv/bin/activate

echo "Virtual environment activated!"
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"
echo ""
echo "To deactivate, run: deactivate"
echo "To install new packages: pip install <package_name>"
echo "To run the application: python quickstart.py" 
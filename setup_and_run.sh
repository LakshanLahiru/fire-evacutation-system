#!/bin/bash
# setup_and_run.sh - Linux/Mac shell script to setup and run the Re-ID system

echo ""
echo "============================================================"
echo " Multi-Video Person Re-ID System - Setup and Launch"
echo "============================================================"
echo ""

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "[1/4] Checking Python version..."
python3 --version
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "[2/4] Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "[2/4] Virtual environment already exists."
fi
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "[3/4] Installing dependencies..."
echo "This may take a few minutes on first run..."
echo ""
pip install --upgrade pip
pip install -r requirements.txt

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

echo ""
echo "Dependencies installed successfully!"
echo ""

# Create necessary directories
echo "[4/4] Creating directories..."
mkdir -p uploads
mkdir -p AI_models
mkdir -p model_data
echo "Directories created."
echo ""

# Start the server
echo "============================================================"
echo " Server is starting..."
echo ""
echo " Once started, open your browser and go to:"
echo " http://localhost:8000/"
echo ""
echo " Press Ctrl+C to stop the server"
echo "============================================================"
echo ""

python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000






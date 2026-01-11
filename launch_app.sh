#!/bin/bash

# NBA Fantasy Optimizer - Local Launcher
# Double-click this file or run: ./launch_app.sh

echo "🏀 NBA Fantasy Lineup Optimizer"
echo "================================"
echo ""

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

echo "📂 Project directory: $DIR"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
else
    echo "✅ Virtual environment found"
    source venv/bin/activate
fi

echo ""
echo "🔍 Checking Kaggle credentials..."

# Check for Kaggle credentials
if [ -f "$HOME/.kaggle/kaggle.json" ]; then
    echo "✅ Kaggle credentials found at ~/.kaggle/kaggle.json"
elif [ ! -z "$KAGGLE_USERNAME" ] && [ ! -z "$KAGGLE_KEY" ]; then
    echo "✅ Kaggle credentials found in environment variables"
else
    echo "⚠️  Kaggle credentials not found (optional)"
    echo ""
    echo "To enable Kaggle data updates:"
    echo "1. Go to https://www.kaggle.com/settings/account"
    echo "2. Click 'Create New API Token'"
    echo "3. Download kaggle.json"
    echo "4. Run: mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/"
    echo ""
    echo "You can still use the app without Kaggle credentials!"
fi

echo ""
echo "🚀 Starting NBA Fantasy Optimizer..."
echo ""
echo "The app will open in your browser at:"
echo "👉 http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the app"
echo ""
echo "================================"
echo ""

# Start Streamlit
streamlit run app.py


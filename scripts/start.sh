#!/bin/bash

# ============================================================================
# Quick Start Script for Active Learning Classifier
# ============================================================================

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║    Active Learning Image Classifier - Quick Start             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✓ Python found: $(python3 --version)"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ pip upgraded"
echo ""

# Install dependencies
echo "📥 Installing dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt --quiet
    echo "✓ Dependencies installed"
else
    echo "❌ requirements.txt not found!"
    exit 1
fi
echo ""

# Optional: Install development dependencies
read -p "Install development dependencies? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "📥 Installing development dependencies..."
    pip install -e ".[dev]" --quiet
    echo "✓ Development dependencies installed"
fi
echo ""

# Check if TensorFlow can use GPU
echo "🔍 Checking GPU availability..."
python3 -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'GPUs Available: {len(gpus)}'); [print(f'  - {gpu.name}') for gpu in gpus]" 2>/dev/null || echo "⚠️  No GPU detected (will use CPU)"
echo ""

# Run tests (optional)
read -p "Run tests before starting? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🧪 Running tests..."
    pytest tests/ -v --tb=short
    echo ""
fi

# Start the application
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║    Starting Streamlit Application...                          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "📍 The application will open in your default browser"
echo "🌐 URL: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the application"
echo ""

streamlit run app.py

# Run the application
echo ""
echo "✅ Setup complete! Launching Streamlit app..."
echo ""
streamlit run app.py

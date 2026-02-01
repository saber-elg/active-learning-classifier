@echo off
REM ============================================================================
REM Quick Start Script for Active Learning Classifier (Windows)
REM ============================================================================

echo ╔════════════════════════════════════════════════════════════════╗
echo ║    Active Learning Image Classifier - Quick Start             ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org/
    pause
    exit /b 1
)

echo ✓ Python found
python --version
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
    echo ✓ Virtual environment created
) else (
    echo ✓ Virtual environment already exists
)
echo.

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat
echo ✓ Virtual environment activated
echo.

REM Upgrade pip
echo ⬆️  Upgrading pip...
python -m pip install --upgrade pip --quiet
echo ✓ pip upgraded
echo.

REM Install dependencies
echo 📥 Installing dependencies...
if exist "requirements.txt" (
    pip install -r requirements.txt --quiet
    echo ✓ Dependencies installed
) else (
    echo ❌ requirements.txt not found!
    pause
    exit /b 1
)
echo.

REM Optional: Install development dependencies
set /p INSTALL_DEV="Install development dependencies? (y/N): "
if /i "%INSTALL_DEV%"=="y" (
    echo 📥 Installing development dependencies...
    pip install -e ".[dev]" --quiet
    echo ✓ Development dependencies installed
    echo.
)

REM Check GPU availability
echo 🔍 Checking GPU availability...
python -c "import tensorflow as tf; gpus = tf.config.list_physical_devices('GPU'); print(f'GPUs Available: {len(gpus)}'); [print(f'  - {gpu.name}') for gpu in gpus]" 2>nul || echo ⚠️  No GPU detected (will use CPU)
echo.

REM Optional: Run tests
set /p RUN_TESTS="Run tests before starting? (y/N): "
if /i "%RUN_TESTS%"=="y" (
    echo 🧪 Running tests...
    pytest tests/ -v --tb=short
    echo.
)

REM Start the application
echo ╔════════════════════════════════════════════════════════════════╗
echo ║    Starting Streamlit Application...                          ║
echo ╚════════════════════════════════════════════════════════════════╝
echo.
echo 📍 The application will open in your default browser
echo 🌐 URL: http://localhost:8501
echo.
echo Press Ctrl+C to stop the application
echo.

streamlit run app.py

pause

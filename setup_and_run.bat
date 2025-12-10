@echo off
REM setup_and_run.bat - Windows batch script to setup and run the Re-ID system
REM For Linux/Mac, use setup_and_run.sh instead

echo.
echo ============================================================
echo  Multi-Video Person Re-ID System - Setup and Launch
echo ============================================================
echo.

REM Check Python installation
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

echo [1/4] Checking Python version...
python --version
echo.

REM Check if requirements are installed
echo [2/4] Installing dependencies...
echo This may take a few minutes on first run...
echo.
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo.
echo Dependencies installed successfully!
echo.

REM Create necessary directories
echo [3/4] Creating directories...
if not exist "uploads" mkdir uploads
if not exist "AI_models" mkdir AI_models
if not exist "model_data" mkdir model_data
echo Directories created.
echo.

REM Start the server
echo [4/4] Starting the Re-ID server...
echo.
echo ============================================================
echo  Server is starting...
echo  
echo  Once started, open your browser and go to:
echo  http://localhost:8000/
echo  
echo  Press Ctrl+C to stop the server
echo ============================================================
echo.

python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000

pause


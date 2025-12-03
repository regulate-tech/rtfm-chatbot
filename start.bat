@echo off
TITLE NHS Chatbot Cookbook - Launcher
CLS

echo ========================================================
echo   NHS CHATBOT COOKBOOK - LAUNCHER
echo ========================================================
echo.

:: 1. Check Python
python --version >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Python is not installed.
    echo Please install Python 3.10+ from python.org
    pause
    exit /b
)

:: 2. Check/Create Venv
IF NOT EXIST ".venv" (
    echo [INFO] Creating virtual environment...
    python -m venv .venv
)

:: 3. Install Dependencies
echo [INFO] Updating dependencies...
call .venv\Scripts\activate
pip install -r requirements.txt --quiet --disable-pip-version-check

:: 4. Check Ollama
where ollama >nul 2>&1
IF %ERRORLEVEL% NEQ 0 (
    echo [WARNING] Ollama not found!
    echo Please install from https://ollama.com
)

:: 5. Run App
echo.
echo [INFO] Starting Application...
streamlit run app-book-v4.py

pause
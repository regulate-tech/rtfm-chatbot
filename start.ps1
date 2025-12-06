<#
.SYNOPSIS
    NHS Chatbot Cookbook - PowerLauncher
.DESCRIPTION
    Checks environment, installs missing libs, and runs v4 app.
#>

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  NHS CHATBOT COOKBOOK - AUTOMATED STARTUP" -ForegroundColor Cyan
Write-Host "========================================================"
Write-Host ""

# 1. Check Python Availability
try {
    $null = python --version
}
catch {
    Write-Error "Python not found! Please install from python.org"
    Read-Host "Press Enter to exit..."
    exit
}

# 2. Virtual Environment Logic
$venvPath = ".\.venv"
$pythonExe = "$venvPath\Scripts\python.exe"

if (-not (Test-Path $venvPath)) {
    Write-Host "[INIT] Creating safe virtual environment (.venv)..." -ForegroundColor Yellow
    python -m venv .venv
}

# 3. Dependency Management
# We run pip using the venv python directly. This is safer than 'activating' in scripts.
Write-Host "[INFO] Verifying libraries..." -ForegroundColor Green
& $pythonExe -m pip install -r requirements.txt --quiet --disable-pip-version-check

# 4. Ollama Check
if (-not (Get-Command "ollama" -ErrorAction SilentlyContinue)) {
    Write-Warning "Ollama is not installed."
    Write-Host "Please download it from https://ollama.com/download" -ForegroundColor Yellow
    # We don't exit, we let them try running anyway (e.g. if using remote server)
}

# 5. Launch Application
Write-Host ""
Write-Host "[LAUNCH] Starting 'The Full Monty'..." -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop." -ForegroundColor Gray
Write-Host ""

& $pythonExe -m streamlit run app_book_v4.py


Read-Host "Application closed. Press Enter to exit..."

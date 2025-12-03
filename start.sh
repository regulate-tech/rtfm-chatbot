#!/bin/bash

# ==========================================
#  NHS CHATBOT COOKBOOK - LAUNCHER (Mac/Linux)
# ==========================================

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}   NHS AI COOKBOOK - INITIALIZATION     ${NC}"
echo -e "${BLUE}========================================${NC}"

# 1. CHECK PYTHON
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}[ERROR] Python 3 not found.${NC}"
    echo "Please install Python 3.10+."
    exit 1
fi

# 2. CHECK VENV MODULE (Common issue on Ubuntu/Debian)
# We try to run a tiny python command to see if venv module exists
python3 -c "import venv" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${RED}[ERROR] The 'venv' module is missing.${NC}"
    echo "On Ubuntu/Debian, please run: sudo apt-get install python3-venv"
    exit 1
fi

# 3. SETUP VIRTUAL ENVIRONMENT
if [ ! -d ".venv" ]; then
    echo -e "${BLUE}[INFO] Creating Virtual Environment (.venv)...${NC}"
    python3 -m venv .venv
    echo -e "${GREEN}[SUCCESS] Environment created.${NC}"
fi

# 4. ACTIVATE & INSTALL
source .venv/bin/activate

echo -e "${BLUE}[INFO] Checking dependencies...${NC}"
pip install --upgrade pip --quiet
# We perform a quiet install; it will skip if already satisfied
pip install -r requirements.txt --quiet --disable-pip-version-check

# 5. CHECK OLLAMA
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}[WARNING] Ollama is not installed.${NC}"
    echo "Please install it from https://ollama.com or run:"
    echo "curl -fsSL https://ollama.com/install.sh | sh"
else
    # Check if Ollama is running
    if ! pgrep -x "ollama" > /dev/null; then
        echo -e "${BLUE}[INFO] Starting Ollama server in background...${NC}"
        ollama serve > /dev/null 2>&1 &
        sleep 2
    fi
fi

# 6. LAUNCH APP
echo ""
echo -e "${GREEN}[START] Launching The Full Monty (v4)...${NC}"
echo "Press Ctrl+C to stop."
echo ""

# Run the final version
streamlit run app-book-v4.py
@echo off
setlocal
cd /d "%~dp0"
set "PYTHON_EXE=python"
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"
echo SRT-M15 local conversation mode (Ollama Qwen3 4B Instruct)
echo Make sure Ollama is installed and you have run: ollama pull qwen3:4b-instruct
"%PYTHON_EXE%" start_local.py --chat-backend ollama --ollama-model qwen3:4b-instruct
if errorlevel 1 (
  echo.
  echo Failed to start. Check Python, torch, and Ollama installation.
  pause
)
endlocal

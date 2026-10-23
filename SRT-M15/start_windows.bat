@echo off
setlocal
cd /d "%~dp0"
set PYTHONUTF8=1
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" start_local.py
) else (
  python start_local.py
)
if errorlevel 1 (
  echo.
  echo Failed to start. Install Python 3.10+ and dependencies:
  echo python -m pip install -r requirements.txt
  pause
)
endlocal

@echo off
setlocal

set "ROOT=%~dp0"
set "VENV_PY=%ROOT%.venv\Scripts\python.exe"
set "FRONTEND_DIR=%ROOT%frontend\chatbot"

if not exist "%VENV_PY%" (
  echo Python launcher not found at "%VENV_PY%".
  echo Create the virtual environment first.
  exit /b 1
)

if not exist "%FRONTEND_DIR%" (
  echo Frontend folder not found at "%FRONTEND_DIR%".
  exit /b 1
)

start "Chatbot Backend" cmd /k "cd /d "%ROOT%" && "%VENV_PY%" -m uvicorn src.web_api:app --reload --host 127.0.0.1 --port 8000"
start "Chatbot Frontend" cmd /k "cd /d "%FRONTEND_DIR%" && npm run dev"

echo Started backend on http://127.0.0.1:8000 and frontend on http://localhost:5173
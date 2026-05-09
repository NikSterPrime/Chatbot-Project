$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$backendPython = Join-Path $root '.venv\Scripts\python.exe'
$frontendDir = Join-Path $root 'frontend\chatbot'

if (-not (Test-Path $backendPython)) {
  throw "Python launcher not found at $backendPython. Create the virtual environment first."
}

if (-not (Test-Path $frontendDir)) {
  throw "Frontend folder not found at $frontendDir."
}

Start-Process -FilePath 'powershell.exe' -ArgumentList @(
  '-NoExit',
  '-ExecutionPolicy', 'Bypass',
  '-Command',
  "Set-Location `"$root`"; & `"$backendPython`" -m uvicorn src.web_api:app --reload --host 127.0.0.1 --port 8000"
)

Start-Process -FilePath 'powershell.exe' -ArgumentList @(
  '-NoExit',
  '-ExecutionPolicy', 'Bypass',
  '-Command',
  "Set-Location `"$frontendDir`"; npm run dev"
)

Write-Host 'Started backend on http://127.0.0.1:8000 and frontend on http://localhost:5173'
#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND_DIR="$ROOT_DIR/frontend/chatbot"
VENV_DIR="$ROOT_DIR/.venv"
PYTHON_BIN="$VENV_DIR/bin/python"
PIP_BIN="$VENV_DIR/bin/pip"
BACKEND_HOST="127.0.0.1"
BACKEND_PORT="8000"

usage() {
  cat <<'EOF'
Usage:
  ./start-chatbot-mac.sh            # install deps (if needed) and run app
  ./start-chatbot-mac.sh all        # same as default
  ./start-chatbot-mac.sh install    # install backend + frontend deps only
  ./start-chatbot-mac.sh run        # run app only (expects deps already installed)
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: '$1' is required but not installed."
    exit 1
  fi
}

install_backend() {
  require_cmd python3

  if [[ ! -d "$VENV_DIR" ]]; then
    echo "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
  fi

  echo "Installing Python dependencies..."
  "$PIP_BIN" install --upgrade pip
  "$PIP_BIN" install -r "$ROOT_DIR/requirements.txt"
}

install_frontend() {
  require_cmd npm

  echo "Installing frontend dependencies..."
  (cd "$FRONTEND_DIR" && npm install)
}

run_app() {
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "Error: Python virtual environment not found at $PYTHON_BIN"
    echo "Run './start-chatbot-mac.sh install' first."
    exit 1
  fi

  if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
    echo "Warning: frontend dependencies are missing. Running npm install first..."
    install_frontend
  fi

  echo "Starting backend on http://$BACKEND_HOST:$BACKEND_PORT"
  (
    cd "$ROOT_DIR"
    "$PYTHON_BIN" -m uvicorn src.web_api:app --reload --host "$BACKEND_HOST" --port "$BACKEND_PORT"
  ) &
  BACKEND_PID=$!

  cleanup() {
    if ps -p "$BACKEND_PID" >/dev/null 2>&1; then
      echo "Stopping backend..."
      kill "$BACKEND_PID" >/dev/null 2>&1 || true
    fi
  }

  trap cleanup EXIT INT TERM

  echo "Starting frontend on http://localhost:5173"
  (
    cd "$FRONTEND_DIR"
    npm run dev
  )
}

MODE="${1:-all}"

case "$MODE" in
  install)
    install_backend
    install_frontend
    echo "Install complete. Run './start-chatbot-mac.sh run' to start the app."
    ;;
  run)
    run_app
    ;;
  all)
    install_backend
    install_frontend
    run_app
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "Unknown option: $MODE"
    usage
    exit 1
    ;;
esac

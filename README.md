# Chatbot Project

Context-aware intent chatbot (TF-IDF + Logistic Regression) with a React + Vite frontend and a FastAPI backend.

This repository contains a local intent classifier and a small web UI that shows session-aware behavior (it can remember a name, keep a short session summary, and run a simple reminder flow). The project optionally integrates a Gemini/LLM summarizer — controlled via `src/.env`.

--

## Quick Links

- Backend: FastAPI app at `src/web_api.py` (default: http://127.0.0.1:8000)
- Frontend: Vite React app at `frontend/chatbot` (default: http://localhost:5173)
- Launchers added to the repo root: `start-chatbot.bat`, `start-chatbot.ps1`, and `start-chatbot-mac.sh`

--

## Requirements

- Python 3.10+ (project uses a virtual environment at `.venv`)
- Node.js + npm (for running the Vite frontend)
- Optional: a Gemini / Google generative API key if you want LLM summarization

## Setup

1. Create and activate the virtual environment (Windows PowerShell):

```powershell
python -m venv .venv
. .venv\Scripts\Activate.ps1
```

2. Install Python dependencies:

```powershell
pip install -r requirements.txt
```

3. Install frontend dependencies:

```powershell
cd frontend\chatbot
npm install
cd ../..
```

4. (Optional) Add your Gemini/Google API key to `src/.env`:

```
# copy src/.env.example (or edit src/.env) and set your values
GOOGLE_API_KEY=your_api_key_here
```

Important: `src/.env` is listed in `.gitignore` and should never be committed. See Security notes below.

--

## Run (single command)

From the repository root you can start both backend and frontend together using the provided Windows launchers:

- Double-click `start-chatbot.bat` (recommended on Windows), or run from a terminal:

```cmd
start-chatbot.bat
```

- Or run the PowerShell launcher:

```powershell
powershell -ExecutionPolicy Bypass -File .\start-chatbot.ps1
```

Both launchers open two console windows: one running the FastAPI backend using the project virtual environment Python, and one running the Vite frontend.

For macOS (Terminal):

```bash
chmod +x ./start-chatbot-mac.sh
./start-chatbot-mac.sh
```

The macOS launcher can also be run as `./start-chatbot-mac.sh install` (install dependencies only) or `./start-chatbot-mac.sh run` (run only).

## Run (manual)

If you prefer to run each piece separately:

Start backend (from repo root, with venv activated):

```powershell
.venv\Scripts\Activate.ps1
python -m uvicorn src.web_api:app --reload --host 127.0.0.1 --port 8000
```

Start frontend (from `frontend/chatbot`):

```powershell
cd frontend\chatbot
npm run dev
```

Open the frontend URL (usually http://localhost:5173) and chat. The UI proxies `/api` requests to `http://127.0.0.1:8000` via Vite config.

--

## Commands (available from both CLI and frontend)

The bot accepts slash commands. These are handled both in the interactive CLI and via the web UI:

- `/help` or `/commands` — show help
- `/about` — info about the bot
- `/memory` — short session memory summary
- `/memory_full` — condensed history + recent turns
- `/use_llm_summary` — enable LLM summarization (requires valid `GOOGLE_API_KEY` in `src/.env`)
- `/use_llm_summary off` — disable LLM summarization
- `/summarize_now` — ask the LLM to summarize current turns
- `/clear` — clear session memory

If the frontend responds that the key is missing, confirm `src/.env` contains `GOOGLE_API_KEY` and restart the backend so the FastAPI process loads the new environment.

--

## Security & Git

- `src/.env` contains private API keys and should NOT be published. This project includes `src/.env` in `.gitignore`.
- The repository previously tracked the virtual environment; do not commit `.venv/`. If `.venv` is present in git history, remove it from the index and add it to `.gitignore`:

```powershell
git rm --cached src/.env || true
git rm -r --cached .venv || true
echo "src/.env" >> .gitignore
echo ".venv/" >> .gitignore
git add .gitignore
git commit -m "Ignore local secrets and venv"
```

If a secret (like an API key) was ever pushed to a public remote, rotate the key immediately with the provider (Google Cloud console) and purge the secret from git history using tools like `git-filter-repo` or the BFG Repo-Cleaner. After a history rewrite, all collaborators must re-clone.

--

## Files of note

- `src/web_api.py` — FastAPI app serving `/api/chat` and session memory endpoints
- `src/chatbot.py` — core chatbot logic; command handling is now shared between CLI and API paths
- `frontend/chatbot` — React + Vite frontend; see `vite.config.js` for the `/api` proxy
- `start-chatbot.bat` and `start-chatbot.ps1` — launchers that start backend (using `.venv\Scripts\python.exe`) and frontend together

--

## Development notes

- The backend exposes a `/api/health` endpoint for quick checks.
- The project uses package-safe relative imports so you can run the API as `python -m uvicorn src.web_api:app` and still run `python src/main.py` for the CLI chatbot.

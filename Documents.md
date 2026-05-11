# Chatbot Project - Complete Documentation

## Project Overview

This is an **intent-based chatbot** built with **TF-IDF + Logistic Regression**. The chatbot loads training phrases from a JSON intent dataset, preprocesses text, trains an ML classifier, and engages in conversation with safety mechanisms.

It supports **two interfaces**:
- **Terminal** — run `src/main.py` directly for a command-line session
- **Web** — a React/Vite frontend communicates with a FastAPI backend via REST API

Additional capabilities include **LLM fallback via Gemini**, **podcast recommendations**, **persistent session memory**, **LLM-assisted conversation summarization**, and a **query analysis/visualization tool**.

### Key Characteristics
- **Language**: Python 3.10+ (backend), JavaScript/React (frontend)
- **ML Framework**: scikit-learn
- **LLM Integration**: Google Gemini via LangChain (optional, requires API key)
- **Architecture**: Intent classification with session memory + web API
- **Interfaces**: Terminal CLI and React web UI
- **Safety Mechanism**: Confidence and margin thresholds to prevent unreliable predictions

---

## Project Structure

```
Chatbot-Project/
├── README.md                          # Project overview and setup instructions
├── Documents.md                       # This comprehensive documentation
├── requirements.txt                   # Python backend dependencies
├── start-chatbot.bat                  # Windows one-click launcher (cmd)
├── start-chatbot.ps1                  # Windows one-click launcher (PowerShell)
├── start-chatbot-mac.sh               # macOS/Linux one-click launcher (bash)
│
├── data/
│   ├── k_intent.json                  # Primary training dataset (22 intents) — preferred
│   ├── intent.json                    # Fallback training dataset
│   └── session_memory.json            # Persistent chat session memory storage
│
├── src/
│   ├── main.py                        # Terminal entry point — calls chatbot()
│   ├── chatbot.py                     # Core chatbot logic, session memory, LLM integration
│   ├── model.py                       # TF-IDF + Logistic Regression training and prediction
│   ├── preprocessing.py               # Text cleaning and normalization
│   ├── utils.py                       # Intent data loader and mapping helpers
│   ├── web_api.py                     # FastAPI REST backend for web interface
│   ├── podcast_recommender.py         # Podcast recommendation engine (CSV-based)
│   └── query_analysis.py              # Query analysis and visualization tool
│
├── scripts/
│   ├── test_model.py                  # Manual model smoke-tester
│   └── verify_loader.py               # Verifies intent data loads correctly
│
├── Documentation/
│   ├── chatbot.md                     # Inline docs for chatbot.py
│   └── model.md                       # Inline docs for model.py
│
└── frontend/
    └── chatbot/                       # React + Vite web frontend
        ├── index.html                 # HTML shell
        ├── package.json               # Frontend npm dependencies
        ├── vite.config.js             # Vite bundler config
        ├── public/
        │   ├── favicon.svg
        │   └── icons.svg
        └── src/
            ├── main.jsx               # React entry point
            ├── App.jsx                # Main UI component
            ├── App.css                # Component styles
            ├── index.css              # Global styles
            └── styles.css             # Additional styles
```

---

## How It Works

### Terminal Mode Execution Flow
1. **main.py** → Loads `.env`, calls `chatbot()`
2. **chatbot.py** → Main chat loop, handles user input and session memory
3. **utils.py** → Loads intents from JSON, builds exact-match mapping
4. **preprocessing.py** → Cleans and normalizes text
5. **model.py** → Generates predictions with confidence and margin scores
6. **Intent Selection** → Applies thresholds; uses Gemini or local fallback if unsafe
7. **Response Selection** → Randomly selects from intent's response list
8. **Output** → Returns response to user
9. **Loop** → Repeats until user exits

### Web Mode Execution Flow
1. **start-chatbot.bat / .ps1 / .sh** → Starts both backend and frontend processes
2. **web_api.py (FastAPI)** → Serves REST endpoints on `http://127.0.0.1:8000`
3. **frontend/chatbot (React/Vite)** → Served on `http://localhost:5173`, proxied to backend
4. Browser sends `POST /api/chat` → `chat_once()` runs the full intent pipeline
5. Response JSON (including `intent`, `confidence`, `margin`, `source`) is displayed in the UI

### Core Algorithm

```
Input: User message
  ↓
Clean text (lowercase, expand contractions, remove punctuation, trim whitespace)
  ↓
Try exact-match lookup (if exact match found → return mapped intent at confidence 1.0)
  ↓
Vectorize with TF-IDF (unigram + bigram + trigram)
  ↓
Predict with Logistic Regression → get probability distribution over all intents
  ↓
Extract top intent, confidence (top probability), margin (top − second)
  ↓
Check safety: confidence ≥ 0.52 AND margin ≥ 0.10?
  ├─ YES → Return predicted intent + random response
  └─ NO  → Try Gemini LLM fallback
              ├─ Gemini available → return Gemini response (tagged [Answered by Gemini])
              └─ Gemini unavailable → return local fallback + top-3 suggestions
  ↓
Update session memory (turn history, topic, profile)
  ↓
Output response
```

---

## Key Components

### 1. `src/main.py`
- Terminal entry point
- Loads `.env` from the `src/` directory (for `GOOGLE_API_KEY` and `GEMINI_MODEL`)
- Calls `chatbot()` from `chatbot.py`

---

### 2. `src/chatbot.py`
The core of the application. Manages the full conversation pipeline for both terminal and web modes.

**LLM Integration (Gemini via LangChain):**
- `create_gemini_llm()` — creates a `ChatGoogleGenerativeAI` instance using `gemini-2.5-flash` (configurable via `GEMINI_MODEL` env var)
- `llm_generate_fallback_response()` — called when confidence/margin thresholds fail; generates a Gemini response with conversation context included
- `llm_update_summary()` — uses Gemini to maintain a rolling conversation summary (≤200 chars)

**ChatSessionMemory class:**
- `turns` — deque of recent messages (max 12 by default)
- `user_name` — remembered user name
- `favorite_topics` — list of topics the user has mentioned
- `reminders` — list of user-created reminders
- `last_user_intent` / `last_bot_intent` — last detected intents
- `current_topic` — active conversation topic
- `last_entities` — extracted entities from messages
- `pending_task` / `pending_slots` / `pending_missing` — multi-turn slot filling state
- `summary_text` — condensed LLM-generated summary of older conversation history
- Persists to `data/session_memory.json` automatically

**Constants:**
| Constant | Value | Purpose |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.52` | Minimum top-intent probability to trust prediction |
| `MARGIN_THRESHOLD` | `0.10` | Minimum gap between top-2 intent probabilities |
| `FALLBACK_TAG` | `"fallback"` | Intent label used for uncertain predictions |
| `PERSISTENT_MEMORY_PATH` | `data/session_memory.json` | Path for saving/loading session memory |
| `EXIT_WORDS` | `{exit, quit, bye, goodbye, see you, later}` | Words that end the terminal session |

**Commands:**
| Command | Effect |
|---|---|
| `/help` | Display help information |
| `/commands` | List all available commands |
| `/about` | Bot introduction and capabilities |
| `/memory` | Display session memory summary |
| `/memory_full` | Display full turn-by-turn conversation history |
| `/use_llm_summary` | Enable LLM-powered conversation summarization |
| `/summarize_now` | Immediately summarize conversation with LLM |
| `/test_gemini` | Ping Gemini to verify API connection |
| `/recommend` | Trigger podcast recommendation |
| `/clear` | Clear all session memory |

---

### 3. `src/model.py`
Trains the ML model at import time and exposes prediction functions.

**ML Pipeline:**
- **Vectorizer**: `TfidfVectorizer(ngram_range=(1,3), sublinear_tf=True, max_df=0.95)`
- **Classifier**: `LogisticRegression(class_weight="balanced", C=2.0, max_iter=1500, solver="lbfgs")`
- Training completes in ~50–200ms on the current ~140-sample dataset

**Prediction Functions:**
| Function | Returns | Use case |
|---|---|---|
| `predict_intent_details(text)` | `(intent, confidence, margin)` | Primary function used by `chat_once()` |
| `predict_intent_rankings(text, top_k=3)` | `[(intent, score), ...]` | Top-K ranking for suggestions |
| `predict_intent_with_confidence(text)` | `(intent, confidence)` | Simplified call used in scripts |
| `_rank_intents_from_cleaned(text)` | Full sorted `[(intent, score)]` list | Internal helper |

---

### 4. `src/preprocessing.py`
Text cleaning pipeline applied to both training data and runtime input.

**Steps:**
1. Lowercase and strip whitespace
2. Expand contractions (e.g. `"can't"` → `"cannot"`, `"i'm"` → `"i am"`)
3. Remove punctuation
4. Normalize whitespace

**Note:** Stopwords are intentionally NOT removed — they improve classification on short intent phrases.

---

### 5. `src/utils.py`
Loads intent data and builds all lookup structures used by the rest of the system.

**Data loading priority:**
1. `INTENT_FILE` environment variable (if set)
2. `data/k_intent.json` (preferred, if it exists)
3. `data/intent.json` (fallback)

**Exports:**
- `intent_to_responses` — dict mapping intent tag → list of response strings
- `texts` — list of all training patterns
- `labels` — list of corresponding intent tags
- `exact_pattern_to_tag` — dict mapping cleaned pattern string → intent tag (for instant lookup)

Supports both JSON schemas: `tag`/`patterns` and `intent`/`text`.

---

### 6. `src/web_api.py`
FastAPI REST backend. Wraps `chat_once()` for consumption by the React frontend.

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Health check — returns `{"status": "ok"}` |
| `GET` | `/api/podcast-filters` | Returns available genre/day/time filter options |
| `POST` | `/api/chat` | Send a message; returns full response with metadata |
| `POST` | `/api/reset/{session_id}` | Clear session memory for a given session |

**Request body (`/api/chat`):**
```json
{ "message": "hello", "session_id": "optional-uuid" }
```

**Response body:**
```json
{
  "session_id": "uuid",
  "response": "Hi! How can I help?",
  "intent": "greeting",
  "confidence": 1.0,
  "margin": 1.0,
  "source": "local"
}
```

`source` is one of: `"local"`, `"local_fallback"`, `"gemini"`.

Sessions are stored in-memory (dict keyed by UUID). The frontend persists `session_id` in `localStorage`.

**CORS:** Allows `http://localhost:5173` and `http://127.0.0.1:5173` (Vite dev server).

**Start backend manually:**
```bash
uvicorn src.web_api:app --reload --host 127.0.0.1 --port 8000
```

---

### 7. `src/podcast_recommender.py`
Recommends podcasts from a CSV dataset (`data/best_podcast_train.csv`) based on user query preferences.

**Detection:**
- `is_podcast_request(text)` — returns `True` if the input contains keywords like `podcast`, `recommend`, `suggest`, `listen`, `episode`

**Preference extraction from natural language:**
- **Genre** — e.g. `"technology"`, `"true crime"`, `"comedy"` (longest alias match wins)
- **Day** — e.g. `"Monday"`, `"Friday"`
- **Time of day** — e.g. `"morning"`, `"evening"`

**Scoring formula (per episode):**
```
score = (listening_time × 0.50) + (host_popularity × 0.30) + (sentiment × 0.20)
```

**Fallback logic:** If strict filtering (all three preferences) returns no results, the system progressively relaxes constraints until at least some results are found.

**API helper:** `get_filter_options()` returns all unique genres, days, and times in the dataset — consumed by the React frontend to populate dropdown filters.

---

### 8. `src/query_analysis.py`
Analysis and visualization tool for understanding chatbot intent quality and fallback behavior.

**Key class: `QueryAnalyzer`**
- Runs queries through the real `chat_once()` pipeline (not hardcoded samples)
- Records `confidence`, `margin`, `source`, `rankings`, fallback reasons per query
- Uses `AnalysisMemory` (subclass of `ChatSessionMemory`) to avoid modifying persistent user data
- LLM fallback is disabled by default during analysis to avoid unintended API calls

**Outputs:**
- `print_summary()` — full text report with fallback rates, source breakdown, per-query details
- `visualize()` — 4-panel matplotlib chart (source pie, intent bar, fallback rate by intent, confidence vs margin scatter)
- `save_json(path)` / `save_csv(path)` — export raw results

**CLI usage:**
```bash
python src/query_analysis.py                         # run sample queries
python src/query_analysis.py --queries-file my.txt   # custom query file
python src/query_analysis.py --no-show               # skip plot window
python src/query_analysis.py --json out.json         # export results
python src/query_analysis.py --csv out.csv
python src/query_analysis.py --plot chart.png
python src/query_analysis.py --allow-llm             # enable Gemini fallback during analysis
```

---

### 9. `data/k_intent.json` (primary dataset)
The preferred training dataset. Takes priority over `intent.json` when present.

**Schema used:**
```json
{
  "intents": [
    {
      "intent": "IntentName",
      "text": ["pattern1", "pattern2"],
      "responses": ["response1", "response2"],
      "extension": { "function": "", "entities": false, "responses": [] },
      "context": { "in": "", "out": "", "clear": false }
    }
  ]
}
```

**22 intents included:**
`Greeting`, `GreetingResponse`, `CourtesyGreeting`, `CourtesyGreetingResponse`, `CurrentHumanQuery`, `NameQuery`, `RealNameQuery`, `TimeQuery`, `Thanks`, `NotTalking2U`, `UnderstandQuery`, `Shutup`, `Swearing`, `GoodBye`, `CourtesyGoodBye`, `WhoAmI`, `Clever`, `Gossip`, `Jokes`, `PodBayDoor`, `PodBayDoorResponse`, `SelfAware`

---

### 10. `data/intent.json` (fallback dataset)
Original fallback dataset. Used only if `k_intent.json` is not present or `INTENT_FILE` env var is not set. Uses `tag`/`patterns`/`responses` schema.

---

### 11. `data/session_memory.json`
Persistent storage of chat session state across restarts.

**Saved fields:** `user_name`, `favorite_topics`, `reminders`, `summary_text`

Loaded on `ChatSessionMemory.__init__()`, saved on `/clear` and `add_reminder()`.

---

### 12. `frontend/chatbot/` — React + Vite Web UI
Single-page application that communicates with the FastAPI backend.

**Tech stack:** React 18, Vite 5, plain CSS (no component library)

**Features:**
- Real-time chat interface with bot/user message bubbles
- Displays intent, confidence, and source metadata below each bot reply
- Podcast filter panel (genre, day, time dropdowns) — populates from `/api/podcast-filters`
- "Get Recommendations" button builds a natural language query from selected filters
- Download Chat button — exports conversation as a `.txt` file
- Reset Session button — clears server-side session memory and resets UI
- `session_id` persisted in `localStorage` across page reloads
- Keyboard shortcut: `Enter` to send (Shift+Enter for newline)
- Auto-scrolls to latest message

**Start frontend manually:**
```bash
cd frontend/chatbot
npm run dev          # development server at http://localhost:5173
npm run build        # production build
npm run preview      # preview production build
```

---

### 13. `scripts/test_model.py`
Manual smoke-test for the ML model. Runs a hardcoded list of queries through `predict_intent_with_confidence()` and `predict_intent_rankings()`, then prints intent, confidence, top-3 rankings, and a sample response for each.

```bash
python scripts/test_model.py
```

---

### 14. `scripts/verify_loader.py`
Verifies that `utils.py` loads intent data correctly. Prints which data file was selected, how many intents and training texts were loaded, and a sample of text-label pairs.

```bash
python scripts/verify_loader.py
```

---

### 15. `requirements.txt`
Python backend dependencies:

```
langchain
langchain-google-genai
python-dotenv
fastapi
uvicorn
```

scikit-learn and numpy are implicit dependencies of the ML pipeline (install separately or add to requirements if needed).

---

### 16. Startup Scripts

| Script | Platform | What it does |
|---|---|---|
| `start-chatbot.bat` | Windows (cmd) | Starts backend (`uvicorn`) + frontend (`npm run dev`) in two cmd windows |
| `start-chatbot.ps1` | Windows (PowerShell) | Same as `.bat` but as a PowerShell script |
| `start-chatbot-mac.sh` | macOS / Linux | Creates `.venv`, installs all deps, starts both processes; supports `install`, `run`, `all` subcommands |

All scripts expect a `.venv` virtual environment at the repo root.

---

### 17. `Documentation/`
Per-file markdown documentation (separate from this file).

| File | Covers |
|---|---|
| `Documentation/chatbot.md` | Dependencies, purpose, inputs/outputs for `chatbot.py` |
| `Documentation/model.md` | Dependencies, inputs/outputs for `model.py` |

---

## Safety & Confidence Thresholds

The chatbot uses **dual safety signals** before trusting a prediction:

### 1. Confidence Score
- **Definition**: Probability of top-ranked intent from `predict_proba()`
- **Threshold**: `0.52`
- **Purpose**: Ensure the model is confident enough about the prediction

### 2. Margin Score
- **Definition**: Difference between top and second-best class probabilities
- **Calculation**: `top_prob − second_best_prob`
- **Threshold**: `0.10`
- **Purpose**: Ensure the top intent is clearly distinct from the runner-up

### Fallback Behavior
If either threshold is **NOT** met:
1. Try **Gemini LLM** (`llm_generate_fallback_response`) — uses recent conversation context
2. If Gemini is unavailable → use **local fallback** response + top-3 intent suggestions
3. Response is tagged with `source: "gemini"` or `source: "local_fallback"` accordingly

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm
- (Optional) Google Gemini API key for LLM fallback and summarization

### 1. Create and activate virtual environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 2. Install Python dependencies
```bash
pip install -r requirements.txt
pip install numpy scikit-learn   # if not pulled in automatically
```

### 3. Set up environment variables (optional — for Gemini)
Create `src/.env`:
```
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash
```
Without this file, the chatbot runs fully locally — Gemini fallback and LLM summarization are silently disabled.

### 4. Install frontend dependencies
```bash
cd frontend/chatbot
npm install
```

### 5. Run

**Terminal mode:**
```bash
python src/main.py
```

**Web mode (manual):**
```bash
# Terminal 1 — backend
uvicorn src.web_api:app --reload --host 127.0.0.1 --port 8000

# Terminal 2 — frontend
cd frontend/chatbot && npm run dev
```

**Web mode (one-click):**
```bash
# Windows (cmd)
start-chatbot.bat

# Windows (PowerShell)
.\start-chatbot.ps1

# macOS/Linux
./start-chatbot-mac.sh
```

Then open `http://localhost:5173` in a browser.

---

## Data Flow Diagram

```
                        ┌────────────────────────────────────┐
                        │          User Input                 │
                        │  (terminal stdin or HTTP POST)      │
                        └──────────────┬─────────────────────┘
                                       │
                              clean_texts()
                         (lowercase, contractions,
                          punctuation, whitespace)
                                       │
                          ┌────────────▼───────────┐
                          │  Exact match lookup?    │
                          │  exact_pattern_to_tag   │
                          └──── YES ──── NO ────────┘
                               │           │
                          confidence=1.0   │
                               │    TF-IDF vectorize
                               │    + LogReg predict_proba
                               │           │
                               │    Extract intent, confidence, margin
                               │           │
                               │    confidence ≥ 0.52
                               │    AND margin ≥ 0.10?
                               │      │         │
                               │     YES        NO
                               │      │         │
                               │      │   Gemini available?
                               │      │     │       │
                               │      │    YES      NO
                               │      │     │       │
                               │      │  Gemini  local fallback
                               │      │  response + suggestions
                               └──────┴────────────┘
                                       │
                              Update session memory
                          (turns, topic, profile, summary)
                                       │
                              Return response + metadata
                          (intent, confidence, margin, source)
```

---

## Session Memory System

### Features
- **Conversation History**: Rolling deque of up to 12 recent turns
- **User Profile**: Tracks user name and favorite topics
- **Reminders**: User-created reminders with timestamps
- **Intent Tracking**: Remembers last user and bot intents
- **Topic Awareness**: Maintains current conversation topic
- **Persistent Storage**: Saves `user_name`, `favorite_topics`, `reminders`, `summary_text` to `data/session_memory.json`
- **Slot Filling**: Multi-turn dialog with `pending_task` / `pending_slots`
- **LLM Summarization**: Optional — condenses older turns into a short summary string using Gemini (`/use_llm_summary` or `/summarize_now`)

### Memory Summary Fields
The `summary()` method returns a `|`-separated string of:
saved turn count, remembered name, favorite topics, last intents, current topic, reminder count, pending task, condensed history

---

## Performance Tuning

### Current Configuration
- **TF-IDF**: `ngram_range=(1,3)`, `sublinear_tf=True`, `max_df=0.95`
- **LogReg**: `class_weight="balanced"`, `C=2.0`, `max_iter=1500`, `solver="lbfgs"`
- **Thresholds**: `confidence=0.52`, `margin=0.10`
- **Training time**: ~50–200ms at startup (~140 samples, 22 intents)

### Optimization Tips
1. **Expand intent coverage**: Add more diverse patterns to underrepresented intents
2. **Keep intents distinct**: Avoid overlapping patterns across tags after preprocessing
3. **Remove duplicates**: Check for patterns that clean to the same string
4. **Tune thresholds**: Use `query_analysis.py` to visualize where predictions fall relative to thresholds
5. **Monitor fallbacks**: High fallback rate = intent gap; add patterns or new intents
6. **Do NOT remove stopwords**: They improve classification on short phrases

---

## Troubleshooting

### Issue: High fallback rate
- **Cause**: Patterns too similar or not representative of actual user input
- **Solution**: Review `k_intent.json` for overlap; add more natural phrasings; run `query_analysis.py`

### Issue: Wrong intent predictions
- **Cause**: Overlapping patterns after preprocessing, or insufficient training data
- **Solution**: Increase margin threshold; add more distinct patterns

### Issue: Gemini not responding
- **Cause**: Missing or invalid `GOOGLE_API_KEY` in `src/.env`
- **Solution**: Run `/test_gemini` in the chatbot to verify connection; check API key

### Issue: Web frontend can't connect to backend
- **Cause**: Backend not running, or running on wrong port
- **Solution**: Ensure `uvicorn` is running on port 8000; check for firewall/port conflicts

### Issue: `ConvergenceWarning` from scikit-learn
- **Cause**: `max_iter` too low for the dataset
- **Solution**: Already set to 1500; increase further if dataset grows significantly

### Issue: Commands not recognized
- **Cause**: Typo or missing `/` prefix
- **Solution**: Type `/help` to see exact command syntax

---

## Dependencies & Versions

### Backend (Python)
| Package | Purpose |
|---|---|
| `numpy` | Numerical operations in model |
| `scikit-learn` | TF-IDF vectorizer and Logistic Regression |
| `langchain` | LLM orchestration layer |
| `langchain-google-genai` | Google Gemini integration |
| `python-dotenv` | Load `.env` API keys |
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server for FastAPI |

### Frontend (Node.js)
| Package | Purpose |
|---|---|
| `react` `react-dom` | UI framework |
| `vite` | Development server and bundler |
| `@vitejs/plugin-react` | Vite plugin for React JSX |

---

## Key Design Decisions

1. **Exact match shortcut**: If cleaned input matches a training pattern exactly, skip ML — return intent at confidence 1.0
2. **Dual safety signals**: Confidence + margin prevents both weak and ambiguous predictions from reaching the user
3. **LLM as fallback only**: Gemini is not used for every response — only when the local model isn't confident, keeping latency low for known intents
4. **Real-time training**: Model trains from JSON at import time (~200ms); no saved model files needed, so data changes are always reflected automatically
5. **Session memory**: Conversation state persists across restarts via JSON for user name, topics, reminders, and summary
6. **Balanced classes**: `class_weight="balanced"` compensates for intents with fewer training examples
7. **Separate terminal and web interfaces**: `main.py` for quick local use; `web_api.py` for browser access — both use the same `chat_once()` function

---

## Future Enhancement Ideas

- Add entity extraction (dates, locations) to improve slot filling
- Sentiment analysis per turn
- Support multiple languages
- Database backend for conversation logging and analytics
- Evaluation metrics dashboard (accuracy, confusion matrix per intent)
- A/B testing framework for threshold tuning
- Expand podcast dataset and add user preference learning
- Add more intent coverage (weather, math, calendar)

---

**Last Updated**: July 2025
**Project Status**: Active Development

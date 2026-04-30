# Chatbot Project - Complete Documentation

## Project Overview

This is an **intent-based terminal chatbot** built with **TF-IDF + Logistic Regression**. The chatbot loads training phrases from a JSON intent dataset, preprocesses text, trains an ML classifier, and engages in a conversational chat loop with safety mechanisms.

### Key Characteristics
- **Language**: Python 3
- **ML Framework**: scikit-learn
- **Architecture**: Intent classification with session memory
- **Interface**: Command-line terminal chatbot
- **Safety Mechanism**: Confidence and margin thresholds to prevent unreliable predictions

---

## Project Structure

```
Chatbot-Project/
├── README.md                      # Project overview and setup instructions
├── Documents.md                   # This comprehensive documentation
├── data/
│   ├── intent.json               # Training dataset with intent patterns and responses
│   └── session_memory.json       # Persistent chat session memory storage
└── src/
    ├── main.py                   # Entry point - initializes and runs the chatbot
    ├── chatbot.py                # Core chatbot logic with ChatSessionMemory class
    ├── model.py                  # ML model training and prediction
    ├── preprocessing.py          # Text preprocessing utilities
    └── utils.py                  # Helper functions for loading data and mapping intents
```

---

## How It Works

### Execution Flow
1. **main.py** → Entry point, calls `chatbot()`
2. **chatbot.py** → Main chat loop, handles user input and session memory
3. **utils.py** → Loads intents from JSON, builds exact-match mapping
4. **preprocessing.py** → Cleans and normalizes text
5. **model.py** → Generates predictions with confidence and margin scores
6. **Intent Selection** → Applies thresholds; uses fallback if unsafe
7. **Response Selection** → Randomly selects from intent's response list
8. **Output** → Returns response to user
9. **Loop** → Repeats until user exits

### Core Algorithm

```
Input: User message
  ↓
Clean text (lowercase, expand contractions, remove punctuation, trim whitespace)
  ↓
Try exact-match lookup (if exact match found → return mapped intent)
  ↓
Vectorize with TF-IDF (unigram + bigram + trigram)
  ↓
Predict with Logistic Regression (get probabilities)
  ↓
Extract top intent and margin (top probability - second highest)
  ↓
Check safety: confidence > CONFIDENCE_THRESHOLD AND margin > MARGIN_THRESHOLD?
  ├─ YES → Return predicted intent
  └─ NO → Return "fallback" intent (with optional top-3 suggestions)
  ↓
Select random response from intent's response list
  ↓
Output: Response to user
```

---

## Key Components

### 1. **src/main.py**
- Simple entry point
- Imports and calls `chatbot()` function from `chatbot.py`

### 2. **src/chatbot.py**
**Main Chatbot Class & Constants:**
- **ChatSessionMemory**: Manages conversation history and user profile
  - `turns`: Deque of recent messages (max 12 by default)
  - `user_name`: Remembered user name
  - `favorite_topics`: List of topics user likes
  - `reminders`: List of user reminders
  - `last_user_intent`: Last detected user intent
  - `last_bot_intent`: Last bot response intent
  - `current_topic`: Active conversation topic
  - `last_entities`: Extracted entities from messages
  - `pending_task`: Task awaiting slot filling
  - `pending_slots`: Slot values for multi-turn dialogs

**Constants:**
- `CONFIDENCE_THRESHOLD = 0.52` → Minimum prediction probability
- `MARGIN_THRESHOLD = 0.10` → Minimum gap between top-2 predictions
- `FALLBACK_TAG = "fallback"` → Intent for uncertain predictions
- `EXIT_WORDS = {exit, quit, bye, goodbye, see you, later}`
- `COMMANDS = {/help, /commands, /about, /memory, /clear}`

**Main Function:**
- `chatbot()`: Main chat loop
  - Displays startup banner
  - Reads user input in loop
  - Handles commands (/help, /commands, /about, /memory, /clear)
  - Detects exit conditions
  - Updates session memory with each exchange

### 3. **src/model.py**
**ML Model Configuration:**
- **Vectorizer**: TF-IDF with ngram_range=(1,3), sublinear_tf=True, max_df=0.95
- **Classifier**: Logistic Regression with class_weight="balanced", C=2.0, max_iter=1500, solver="lbfgs"

**Training Process:**
1. Load texts and labels from utils
2. Clean all texts with preprocessing
3. Fit TF-IDF vectorizer on cleaned texts
4. Train Logistic Regression on vectorized data

**Prediction Functions:**
- `predict_intent_details(text)` → Returns (intent, confidence, margin)
- `predict_intent_rankings(text, top_k=3)` → Returns top-K intents with probabilities
- `_rank_intents_from_cleaned(cleaned_text)` → Internal ranking function

### 4. **src/preprocessing.py**
**Text Cleaning Pipeline:**
1. Convert to lowercase and strip whitespace
2. Expand contractions (e.g., "can't" → "cannot", "i'm" → "i am")
3. Remove punctuation
4. Normalize whitespace

**Contraction Dictionary:** 24 common contractions mapped to expanded forms

**Note**: Stopwords are NOT removed (keeps "a", "the", "is", etc.) because they improve classification on short intent phrases.

### 5. **src/utils.py**
**Key Functions:**
- `load_intents()` → Reads intent.json and extracts patterns/responses
- `intent_to_responses(intent_tag)` → Returns list of responses for an intent
- `exact_pattern_to_tag` → Dictionary mapping cleaned patterns to intent tags
- `texts` → List of all training phrases
- `labels` → Corresponding intent tags

### 6. **data/intent.json**
**Structure:**
```json
{
  "intents": [
    {
      "tag": "intent_name",
      "patterns": ["pattern1", "pattern2", ...],
      "responses": ["response1", "response2", ...]
    },
    ...
  ]
}
```

**Example Intents Include:**
- `greeting` - Hello, Hi, Hey, etc.
- `goodbye` - Bye, Goodbye, See you, etc.
- `thanks` - Thank you, Thanks, Appreciate it, etc.
- `affirm` - Yes, Yeah, Sure, etc.
- `deny` - No, Nope, Not at all, etc.
- `ask_name` - What's your name?, Who are you?, etc.
- `ask_time` - What time is it?, Tell me the time, etc.
- `ask_date` - What's today?, What date is it?, etc.
- `help` - Help, Can you help?, I need help, etc.
- `about` - Tell me about yourself, What can you do?, etc.
- `fallback` - I don't understand, Sorry?, etc. (used for low-confidence predictions)

### 7. **data/session_memory.json**
**Purpose**: Persistent storage of chat session state
**Contents**: JSON serialization of ChatSessionMemory object
**Loaded/Saved On**: Chat initialization and explicit commands

---

## Safety & Confidence Thresholds

The chatbot uses **dual safety signals** before trusting a prediction:

### 1. Confidence Score
- **Definition**: Probability of top-ranked intent
- **Threshold**: 0.52 (52%)
- **Purpose**: Ensure model is confident enough about the prediction

### 2. Margin Score
- **Definition**: Difference between top and second-best class probabilities
- **Calculation**: `top_prob - second_best_prob`
- **Threshold**: 0.10 (10%)
- **Purpose**: Ensure top intent is clearly distinct from alternatives

### Fallback Behavior
If either threshold is **NOT** met:
- Response intent is set to "fallback"
- User gets a generic fallback response
- Top-3 ranking suggestions may be provided for better UX

---

## User Commands

The chatbot supports special commands prefixed with `/`:

| Command | Effect |
|---------|--------|
| `/help` | Display help information |
| `/commands` | List all available commands |
| `/about` | Bot introduction and capabilities |
| `/memory` | Display session memory summary |
| `/clear` | Clear all session memory |

**Exit Commands** (case-insensitive):
- `exit`, `quit`, `bye`, `goodbye`, `see you`, `later`

---

## Installation & Setup

### Prerequisites
- Python 3.7+
- pip package manager

### Install Dependencies
```bash
pip install numpy scikit-learn
```

### Run Chatbot
```bash
python src/main.py
```

---

## Performance Tuning

### Current Configuration (Optimized)
- **TF-IDF Settings**: ngram_range=(1,3), sublinear_tf=True, max_df=0.95
- **Model Settings**: class_weight="balanced", C=2.0, max_iter=1500, solver="lbfgs"
- **Thresholds**: Confidence=0.52, Margin=0.10

### Optimization Tips
1. **Improve Intent Coverage**: Expand patterns for underrepresented intents
2. **Keep Intents Distinct**: Avoid overlapping patterns after preprocessing
3. **Remove Duplicates**: Check for duplicate phrases across different intent tags
4. **Add Realistic Phrasings**: Include common user variations
5. **Tune Thresholds**: Adjust confidence/margin thresholds based on real chat data
6. **Monitor Fallbacks**: Track which queries fall back to understand coverage gaps

### Known Best Practices (from repo notes)
- **Do NOT remove stopwords** - they improve classification on short phrases
- **Use smoke testing** on Windows: `"hi`nwhat time is it`nwhat can you do`nexit" | python src/main.py`
- TF-IDF with bigrams/trigrams significantly outperforms unigrams alone
- Balanced class weights help with imbalanced training data

---

## Session Memory System

### Features
- **Conversation History**: Stores up to 12 recent turns (configurable)
- **User Profile**: Tracks user name, favorite topics
- **Reminders**: Can store user-created reminders
- **Intent Tracking**: Remembers last user and bot intents
- **Topic Awareness**: Maintains current conversation topic
- **Persistent Storage**: Saves to JSON file for resume capability
- **Slot Filling**: Supports multi-turn dialogs with pending slots

### Memory Persistence
- Automatically loaded on chat startup from `data/session_memory.json`
- Saved on explicit `/clear` command or at key points
- Users can view summary with `/memory` command

---

## Data Flow Diagram

```
User Input
    ↓
Text Preprocessing (clean_texts)
    ├─ Lowercase
    ├─ Expand Contractions
    ├─ Remove Punctuation
    └─ Normalize Whitespace
    ↓
Exact Match Check? → YES → Return Mapped Intent
    ↓ NO
TF-IDF Vectorization
    ↓
Logistic Regression Prediction
    ↓
Extract Intent, Confidence, Margin
    ↓
Safety Checks (Confidence & Margin Thresholds)
    ├─ PASS → Use Predicted Intent
    └─ FAIL → Use Fallback Intent
    ↓
Select Random Response
    ↓
Update Session Memory
    ↓
Output Response to User
```

---

## Common Intent Examples

### Greetings
- Patterns: hi, hello, hey, hiya, good morning, howdy
- Responses: "Hi there. Great to have you here.", "Hello. How can I help today?"

### Goodbyes
- Patterns: bye, goodbye, see you, take care, farewell
- Responses: "Goodbye. Take care.", "See you soon."

### Questions
- ask_time: "What time is it?", "Tell me the time"
- ask_date: "What's today?", "What date is it?"
- ask_name: "What are you called?", "Who are you?"

### Affirmations/Negations
- affirm: "Yes", "Yeah", "Sure", "Yep"
- deny: "No", "Nope", "Not at all", "Negative"

### Help & Meta
- help: "Help", "Can you help?", "I need assistance"
- about: "Tell me about yourself", "What can you do?"

---

## Model Evaluation Considerations

- **Accuracy Metric**: Not provided; recommend evaluating on held-out test set
- **Common Issues**:
  - Low confidence on ambiguous queries → Triggers fallback (expected behavior)
  - Intent overlap after preprocessing → Reduce margin threshold or improve intent patterns
  - Rare intents → Add more training patterns or use class weighting
  
- **Testing Approach**: Use smoke tests with known phrases to verify pipeline

---

## Troubleshooting

### Issue: High Fallback Rate
- **Cause**: Patterns are too similar or not representative
- **Solution**: Review intent.json for overlap; add more diverse patterns

### Issue: Wrong Intent Predictions
- **Cause**: Overlapping preprocessing results or insufficient training data
- **Solution**: Increase margin threshold; add more distinct training patterns

### Issue: Model Takes Long to Train
- **Cause**: Too many patterns or high max_iter setting
- **Solution**: Reduce max_iter from 1500; consider data sample size

### Issue: Commands Not Recognized
- **Cause**: Typo or missing `/` prefix
- **Solution**: Use `/help` to see exact command syntax

---

## Dependencies & Versions

- **numpy**: For numerical operations in model
- **scikit-learn**: For TF-IDF vectorizer and Logistic Regression
- **Python 3.7+**: Core language requirement

---

## Key Design Decisions

1. **Exact Match Shortcut**: Speed optimization - if input exactly matches cleaned pattern, skip ML prediction
2. **Dual Safety Signals**: Confidence + Margin prevents incorrect predictions while allowing reasonable uncertainty
3. **Session Memory**: Stores conversation state for multi-turn context awareness
4. **Deque-based History**: Efficient memory management with fixed size
5. **Balanced Classes**: Handles imbalanced training data
6. **TF-IDF + Bigrams**: Better feature representation than just unigrams

---

## Future Enhancement Ideas

- Add entity extraction (names, dates, locations)
- Implement context-aware responses using session memory
- Add sentiment analysis
- Support multiple languages
- Database backend for conversation logging
- Web interface (Flask/FastAPI)
- API integration (time, weather, news)
- More sophisticated slot filling for complex intents
- Evaluation metrics and A/B testing framework

---

## Contact & Support

For issues or questions about this project, refer to README.md or examine the inline code comments in src/ files.

---

**Last Updated**: April 2026  
**Project Status**: Active Development

# Chatbot Project

Intent-based terminal chatbot built with TF-IDF + Logistic Regression.

## Overview

This project loads training phrases from a JSON intent dataset, preprocesses text, trains an intent classifier, and responds in a chat loop.

The chatbot uses two safety signals before trusting a prediction:
- Confidence: top class probability
- Margin: difference between top and second-best class probabilities

If either signal is low, the chatbot returns a fallback response.

## Project Structure

```text
Chatbot_Project/
├── README.md
├── Documentation.md
├── data/
│   └── intent.json
└── src/
		├── chatbot.py
		├── main.py
		├── model.py
		├── preprocessing.py
		└── utils.py
```

## How It Works

```text
src/main.py
	-> src/chatbot.py
	-> src/utils.py + data/intent.json
	-> src/preprocessing.py
	-> src/model.py
	-> predict_intent_details()
	-> intent selection
	-> response selection
	-> output
	-> user input
	-> repeat until exit
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install numpy scikit-learn
pip install langchain-google-genai
```

3. Add your Gemini API key to `src/.env`:

```bash
GOOGLE_API_KEY=your_api_key_here
```

The chatbot uses Gemini only for the optional summary helper behind `/use_llm_summary` and `/summarize_now`. The main intent classifier still runs locally.

## Run

From the project root:

```bash
python src/main.py
```

Type messages in the prompt. To quit, type one of:
- exit
- quit
- bye

## Model Details

- Vectorizer: TF-IDF with unigram + bigram + trigram features
- Classifier: Logistic Regression
- Exact-match shortcut: if cleaned input exactly matches a cleaned training pattern, the mapped intent is returned directly
- Fallback logic: used when confidence or margin is below threshold

Current thresholds in src/chatbot.py:
- CONFIDENCE_THRESHOLD = 0.52
- MARGIN_THRESHOLD = 0.10

## Recent Improvements

- Expanded and cleaned intent dataset for broader, less-overlapping coverage
- Added practical intents: `ask_time`, `ask_date`, `compliment`, `affirmation`, `negation`
- Upgraded vectorizer and model settings for better text generalization
- Added top-intent ranking support for smarter fallback suggestions
- Added a more presentable chat UI with startup banner and command support
- Added chat commands: `/help`, `/commands`, `/about`

## Notes for Better Accuracy

- Keep intent patterns distinct after preprocessing.
- Avoid duplicate phrases across different intent tags.
- Add more realistic user phrasings per intent.
- Tune confidence and margin thresholds on real chat examples.

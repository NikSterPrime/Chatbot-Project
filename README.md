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
pip install numpy scikit-learn nltk
```

NLTK stopwords are downloaded automatically by the preprocessing module.

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

- Vectorizer: TF-IDF with unigram + bigram features
- Classifier: Logistic Regression
- Exact-match shortcut: if cleaned input exactly matches a cleaned training pattern, the mapped intent is returned directly
- Fallback logic: used when confidence or margin is below threshold

Current thresholds in src/chatbot.py:
- CONFIDENCE_THRESHOLD = 0.50
- MARGIN_THRESHOLD = 0.08

## Notes for Better Accuracy

- Keep intent patterns distinct after preprocessing.
- Avoid duplicate phrases across different intent tags.
- Add more realistic user phrasings per intent.
- Tune confidence and margin thresholds on real chat examples.

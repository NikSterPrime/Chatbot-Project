import json
from pathlib import Path
from preprocessing import clean_texts

data_path = Path(__file__).resolve().parents[1] / "data" / "intent.json"

with data_path.open(encoding="utf-8") as file:
    data = json.load(file)

intent_to_responses = {
    intent["tag"]: intent["responses"] for intent in data["intents"]
}

texts = []
labels = []
exact_pattern_to_tag = {}

for intent in data["intents"]:
    for pattern in intent["patterns"]:
        key = clean_texts(pattern)

        if not key:
            continue

        # Keep only the first owner for a pattern to avoid contradictory labels.
        if key in exact_pattern_to_tag:
            continue

        exact_pattern_to_tag[key] = intent["tag"]
        texts.append(pattern)
        labels.append(intent["tag"])


import json
import os
import re
from pathlib import Path
from preprocessing import clean_texts


def camel_to_snake(name: str) -> str:
    name = re.sub('(.)([A-Z][a-z]+)', r"\1_\2", name)
    name = re.sub('([a-z0-9])([A-Z])', r"\1_\2", name)
    return name.replace('-', '_').lower()


# Prefer an explicit INTENT_FILE, then k_intent.json, then fallback to intent.json
repo_data_dir = Path(__file__).resolve().parents[1] / "data"
intent_file = os.getenv("INTENT_FILE")
if intent_file:
    data_path = Path(intent_file)
else:
    k_path = repo_data_dir / "k_intent.json"
    fallback = repo_data_dir / "intent.json"
    data_path = k_path if k_path.exists() else fallback

with data_path.open(encoding="utf-8") as file:
    data = json.load(file)


# Small mapping for canonical tags used by runtime code
canonical_tag_map = {
    "time_query": "ask_time",
    "timequery": "ask_time",
    "name_query": "asking_name",
    "namequery": "asking_name",
}

intent_to_responses = {}
texts = []
labels = []
exact_pattern_to_tag = {}

raw_intents = data.get("intents") or data

for intent in raw_intents:
    # Support either schema: 'tag'/'patterns' or 'intent'/'text'
    tag = intent.get("tag") or intent.get("intent")
    patterns = intent.get("patterns") or intent.get("text") or []
    responses = intent.get("responses") or intent.get("extension") or []

    if not tag:
        continue

    normalized = camel_to_snake(tag)
    canonical = canonical_tag_map.get(normalized, normalized)

    intent_to_responses[canonical] = responses

    for pattern in patterns:
        key = clean_texts(pattern)
        if not key:
            continue
        # Keep only the first owner for a pattern to avoid contradictory labels.
        if key in exact_pattern_to_tag:
            continue

        exact_pattern_to_tag[key] = canonical
        texts.append(pattern)
        labels.append(canonical)


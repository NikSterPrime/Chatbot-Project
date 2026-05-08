import sys
sys.path.insert(0, "src")
from model import predict_intent_with_confidence, predict_intent_rankings
from utils import intent_to_responses

tests = [
    "Hello",
    "Hi there",
    "My name is Aditya",
    "What time is it?",
    "What's your name?",
    "Tell me a joke",
    "Shut up",
]

for t in tests:
    intent, conf = predict_intent_with_confidence(t)
    rankings = predict_intent_rankings(t, top_k=3)
    response = intent_to_responses.get(intent, ["<no response>"])[0]
    print(f"INPUT: {t}")
    print(f"  PREDICT: {intent} ({conf:.2f})")
    print(f"  TOP3: {rankings}")
    print(f"  SAMPLE RESPONSE: {response}\n")
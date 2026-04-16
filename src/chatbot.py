import random

from model import predict_intent_details
from utils import intent_to_responses

FALLBACK_TAG = "fallback"
CONFIDENCE_THRESHOLD = 0.50
MARGIN_THRESHOLD = 0.08
EXIT_WORDS = {"exit", "quit", "bye"}

def give_response(intent):
    responses = intent_to_responses.get(intent) or intent_to_responses.get(FALLBACK_TAG, [])
    if not responses:
        return "I am not sure how to respond right now."
    return random.choice(responses)

        
def chatbot():
    print("Chatbot is ready. Type 'exit' to stop.")

    while True:
        user_input = input("$ ")
        if user_input.strip().lower() in EXIT_WORDS:
            return
        
        intent, confidence, margin = predict_intent_details(user_input)

        if confidence < CONFIDENCE_THRESHOLD or margin < MARGIN_THRESHOLD:
            intent = FALLBACK_TAG
        
        response = give_response(intent)
        print(f"> {response}")


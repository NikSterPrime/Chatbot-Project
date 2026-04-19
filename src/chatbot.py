import random
import re
from collections import deque
from datetime import datetime

from model import predict_intent_details, predict_intent_rankings
from utils import intent_to_responses

FALLBACK_TAG = "fallback"
CONFIDENCE_THRESHOLD = 0.52
MARGIN_THRESHOLD = 0.10
EXIT_WORDS = {"exit", "quit", "bye"}
COMMANDS = {"/help", "/commands", "/about", "/memory", "/clear"}


class ChatSessionMemory:
    def __init__(self, max_turns=12):
        self.max_turns = max_turns
        self.turns = deque(maxlen=max_turns)
        self.user_name = None
        self.favorite_topics = []
        self.last_user_intent = None
        self.last_bot_intent = None

    def remember_user(self, text, intent):
        self.turns.append({"role": "user", "text": text, "intent": intent})
        self.last_user_intent = intent
        self._extract_user_profile(text)

    def remember_bot(self, text, intent):
        self.turns.append({"role": "bot", "text": text, "intent": intent})
        self.last_bot_intent = intent

    def clear(self):
        self.turns.clear()
        self.user_name = None
        self.favorite_topics = []
        self.last_user_intent = None
        self.last_bot_intent = None

    def summary(self):
        if not self.turns:
            return "No session memory yet."

        details = [f"Saved turns: {len(self.turns)}"]
        if self.user_name:
            details.append(f"Remembered name: {self.user_name}")
        if self.favorite_topics:
            details.append(f"Favorite topics: {', '.join(self.favorite_topics[:3])}")
        if self.last_user_intent:
            details.append(f"Last user intent: {self.last_user_intent}")
        if self.last_bot_intent:
            details.append(f"Last bot intent: {self.last_bot_intent}")
        return " | ".join(details)

    def _extract_user_profile(self, text):
        lowered = text.strip().lower()

        name_match = re.search(r"\b(?:my name is|call me)\s+([a-zA-Z]+)\b", lowered)
        if name_match:
            self.user_name = name_match.group(1).capitalize()

        topic_match = re.search(r"\b(?:i like|i love|my favorite topic is)\s+([a-zA-Z ]{2,30})$", lowered)
        if topic_match:
            topic = " ".join(topic_match.group(1).split()).strip().lower()
            if topic and topic not in self.favorite_topics:
                self.favorite_topics.append(topic)


def _render_welcome():
    print("=" * 64)
    print(" Chatbot Assistant")
    print(" Type your message and press Enter.")
    print(" Commands: /help, /about, /memory, /clear, exit")
    print("=" * 64)


def _render_help():
    print("Try prompts like:")
    print("- hi")
    print("- what can you do")
    print("- tell me a joke")
    print("- i feel stressed")
    print("- what time is it")
    print("Commands: /help, /about, /memory, /clear, exit")


def _handle_special_intents(intent):
    now = datetime.now()
    if intent == "ask_time":
        return f"Current local time is {now.strftime('%I:%M %p')}"
    if intent == "ask_date":
        return f"Today is {now.strftime('%A, %B %d, %Y')}"
    return None


def _suggest_from_rankings(rankings):
    suggestions = [intent for intent, score in rankings if intent != FALLBACK_TAG and score >= 0.20]
    if not suggestions:
        return ""
    unique = list(dict.fromkeys(suggestions))[:2]
    return f" Maybe you meant: {', '.join(unique)}."


def _contextual_response(intent, memory):
    if intent == "follow_up":
        if memory.last_bot_intent == "jokes":
            return "Want another one? I can keep the jokes coming."
        if memory.last_user_intent == "user_feeling":
            return "If it helps, we can try a quick reset: deep breath, short walk, and one tiny next step."
        if memory.last_user_intent == "asking_capabilities":
            return "I can also remember your name and recent context in this session."

    if intent == "affirmation":
        if memory.last_bot_intent == "help":
            return "Great. Share the exact problem and I will break it into steps."
        return "Nice. What would you like to do next?"

    if intent == "negation":
        if memory.last_bot_intent in {"help", "follow_up"}:
            return "No worries. We can switch to a different topic anytime."

    if intent == "asking_name" and memory.user_name:
        return f"I am Assistant. I remember your name is {memory.user_name}."

    if intent == "casual_talk" and memory.user_name:
        return f"I am doing well, {memory.user_name}. What is on your mind?"

    return None


def _handle_profile_intro(user_input, memory):
    name_match = re.search(r"\b(?:my name is|call me)\s+([a-zA-Z]+)\b", user_input.strip().lower())
    if not name_match:
        return None

    memory.user_name = name_match.group(1).capitalize()
    return f"Nice to meet you, {memory.user_name}. I will remember your name for this session."

def give_response(intent):
    responses = intent_to_responses.get(intent) or intent_to_responses.get(FALLBACK_TAG, [])
    if not responses:
        return "I am not sure how to respond right now."
    return random.choice(responses)

        
def chatbot():
    _render_welcome()
    memory = ChatSessionMemory()

    while True:
        user_input = input("$ ")
        normalized = user_input.strip().lower()

        if normalized in EXIT_WORDS:
            print("> Thanks for chatting. See you next time.")
            return

        if normalized in COMMANDS:
            if normalized == "/about":
                print("> I am an intent-based chatbot powered by TF-IDF + Logistic Regression.")
            elif normalized == "/memory":
                print(f"> {memory.summary()}")
            elif normalized == "/clear":
                memory.clear()
                print("> Session memory cleared.")
            else:
                _render_help()
            continue

        profile_response = _handle_profile_intro(user_input, memory)
        if profile_response:
            memory.remember_user(user_input, "profile_update")
            memory.remember_bot(profile_response, "profile_update")
            print(f"> {profile_response}")
            continue
        
        intent, confidence, margin = predict_intent_details(user_input)
        rankings = predict_intent_rankings(user_input, top_k=3)

        special_response = _handle_special_intents(intent)
        if special_response and confidence >= 0.45:
            memory.remember_user(user_input, intent)
            memory.remember_bot(special_response, intent)
            print(f"> {special_response}")
            continue

        if confidence < CONFIDENCE_THRESHOLD or margin < MARGIN_THRESHOLD:
            intent = FALLBACK_TAG
            response = give_response(intent) + _suggest_from_rankings(rankings)
            memory.remember_user(user_input, intent)
            memory.remember_bot(response, intent)
            print(f"> {response}")
            continue
        
        response = _contextual_response(intent, memory) or give_response(intent)
        memory.remember_user(user_input, intent)
        memory.remember_bot(response, intent)
        print(f"> {response}")


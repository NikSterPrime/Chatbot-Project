import random
import re
import json
from collections import deque
from datetime import datetime
from pathlib import Path

from model import predict_intent_details, predict_intent_rankings
from utils import intent_to_responses

FALLBACK_TAG = "fallback"
CONFIDENCE_THRESHOLD = 0.52
MARGIN_THRESHOLD = 0.10
EXIT_WORDS = {"exit", "quit", "bye", "goodbye", "see you", "later"}
COMMANDS = {"/help", "/commands", "/about", "/memory", "/clear"}
PERSISTENT_MEMORY_PATH = Path(__file__).resolve().parents[1] / "data" / "session_memory.json"


class ChatSessionMemory:
    def __init__(self, max_turns=12):
        self.max_turns = max_turns
        self.turns = deque(maxlen=max_turns)
        self.user_name = None
        self.favorite_topics = []
        self.reminders = []
        self.last_user_intent = None
        self.last_bot_intent = None
        self.current_topic = None
        self.last_entities = {}
        self.pending_task = None
        self.pending_slots = {}
        self.pending_missing = []

        self._load_persistent()

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
        self.reminders = []
        self.last_user_intent = None
        self.last_bot_intent = None
        self.current_topic = None
        self.last_entities = {}
        self.clear_pending()
        self._save_persistent()

    def clear_pending(self):
        self.pending_task = None
        self.pending_slots = {}
        self.pending_missing = []

    def has_pending(self):
        return self.pending_task is not None

    def set_pending(self, task, slots=None, missing=None):
        self.pending_task = task
        self.pending_slots = slots or {}
        self.pending_missing = missing or []

    def set_topic(self, topic, entities=None):
        self.current_topic = topic
        if entities:
            self.last_entities = entities

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
        if self.current_topic:
            details.append(f"Current topic: {self.current_topic}")
        if self.reminders:
            details.append(f"Saved reminders: {len(self.reminders)}")
        if self.pending_task:
            details.append(f"Pending task: {self.pending_task}")
        return " | ".join(details)

    def add_reminder(self, message, when_text):
        reminder = {
            "message": message,
            "when": when_text,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.reminders.append(reminder)
        self.last_entities = {"message": message, "when": when_text}
        self.current_topic = "reminder"
        self._save_persistent()
        return reminder

    def _load_persistent(self):
        if not PERSISTENT_MEMORY_PATH.exists():
            return

        try:
            data = json.loads(PERSISTENT_MEMORY_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return

        self.user_name = data.get("user_name")
        self.favorite_topics = data.get("favorite_topics", [])
        self.reminders = data.get("reminders", [])

    def _save_persistent(self):
        payload = {
            "user_name": self.user_name,
            "favorite_topics": self.favorite_topics,
            "reminders": self.reminders,
        }
        try:
            PERSISTENT_MEMORY_PATH.parent.mkdir(parents=True, exist_ok=True)
            PERSISTENT_MEMORY_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            pass

    def _extract_user_profile(self, text):
        lowered = text.strip().lower()

        name_match = re.search(r"\b(?:my name is|call me)\s+([a-zA-Z]+)\b", lowered)
        if name_match:
            self.user_name = name_match.group(1).capitalize()
            self._save_persistent()

        topic_match = re.search(r"\b(?:i like|i love|my favorite topic is)\s+([a-zA-Z ]{2,30})$", lowered)
        if topic_match:
            topic = " ".join(topic_match.group(1).split()).strip().lower()
            if topic and topic not in self.favorite_topics:
                self.favorite_topics.append(topic)
                self._save_persistent()


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
    print("- remind me to drink water at 7 pm")
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
        if memory.current_topic == "reminder":
            if memory.reminders:
                latest = memory.reminders[-1]
                return (
                    f"Your latest reminder is '{latest['message']}' at {latest['when']}. "
                    "You can ask me to add another one."
                )
            return "We were talking about reminders. You can say: remind me to stretch at 5 pm."
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
    memory._save_persistent()
    return f"Nice to meet you, {memory.user_name}. I will remember your name for this session."


def _extract_reminder_parts(text):
    cleaned = " ".join(text.strip().lower().split())

    message = None
    time_text = None

    message_match = re.search(r"(?:remind me to|remember to)\s+(.+?)(?:\s+(?:at|on|tomorrow|today|in)\b|$)", cleaned)
    if message_match:
        message = message_match.group(1).strip()

    time_match = re.search(r"\b(?:at\s+\d{1,2}(?::\d{2})?\s*(?:am|pm)?|tomorrow|today|in\s+\d+\s+(?:minutes?|hours?|days?))\b", cleaned)
    if time_match:
        time_text = time_match.group(0).strip()

    if not time_text and re.fullmatch(r"\d{1,2}(?::\d{2})?\s*(?:am|pm)", cleaned):
        time_text = cleaned

    return message, time_text


def _normalize_when_text(when_text):
    if not when_text:
        return when_text
    normalized = " ".join(when_text.strip().lower().split())
    return normalized[3:] if normalized.startswith("at ") else normalized


def _start_reminder_flow(user_input, memory):
    lowered = user_input.strip().lower()
    if not re.search(r"\b(remind me|set reminder|remember to)\b", lowered):
        return None

    message, when_text = _extract_reminder_parts(user_input)
    when_text = _normalize_when_text(when_text)

    missing = []
    if not message:
        missing.append("message")
    if not when_text:
        missing.append("when")

    if missing:
        memory.set_pending("set_reminder", slots={"message": message, "when": when_text}, missing=missing)
        if missing[0] == "message":
            return "Sure. What should I remind you about?"
        return "Got it. When should I remind you?"

    memory.add_reminder(message, when_text)
    return f"Reminder saved: '{message}' at {when_text}."


def _continue_pending_flow(user_input, memory):
    if not memory.has_pending():
        return None

    if memory.pending_task != "set_reminder":
        memory.clear_pending()
        return None

    message, when_text = _extract_reminder_parts(user_input)
    when_text = _normalize_when_text(when_text)

    if "message" in memory.pending_missing:
        candidate = message or user_input.strip()
        if candidate:
            memory.pending_slots["message"] = candidate
            memory.pending_missing.remove("message")

    if "when" in memory.pending_missing:
        candidate = when_text
        if candidate:
            memory.pending_slots["when"] = candidate
            memory.pending_missing.remove("when")

    if memory.pending_missing:
        if memory.pending_missing[0] == "message":
            return "I still need the reminder message. What should I remind you about?"
        return "I still need the time. You can say something like 'at 7 pm' or 'tomorrow'."

    saved = memory.add_reminder(memory.pending_slots["message"], memory.pending_slots["when"])
    memory.clear_pending()
    return f"Reminder saved: '{saved['message']}' at {saved['when']}."

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

        pending_response = _continue_pending_flow(user_input, memory)
        if pending_response:
            memory.remember_user(user_input, "pending_input")
            memory.remember_bot(pending_response, "set_reminder")
            print(f"> {pending_response}")
            continue

        profile_response = _handle_profile_intro(user_input, memory)
        if profile_response:
            memory.remember_user(user_input, "profile_update")
            memory.remember_bot(profile_response, "profile_update")
            print(f"> {profile_response}")
            continue

        reminder_response = _start_reminder_flow(user_input, memory)
        if reminder_response:
            memory.remember_user(user_input, "set_reminder")
            memory.remember_bot(reminder_response, "set_reminder")
            print(f"> {reminder_response}")
            continue
        
        intent, confidence, margin = predict_intent_details(user_input)
        rankings = predict_intent_rankings(user_input, top_k=3)

        special_response = _handle_special_intents(intent)
        if special_response and confidence >= 0.45:
            memory.set_topic(intent)
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
        
        if intent not in {"follow_up", "affirmation", "negation", "fallback"}:
            memory.set_topic(intent)

        response = _contextual_response(intent, memory) or give_response(intent)
        memory.remember_user(user_input, intent)
        memory.remember_bot(response, intent)
        print(f"> {response}")


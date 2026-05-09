from pathlib import Path
from dotenv import load_dotenv

# Load local .env (kept in src/.env) so optional API keys become available to the app.
load_dotenv(Path(__file__).resolve().parent / ".env")

try:
	from .chatbot import chatbot
except ImportError:
	from chatbot import chatbot


chatbot()
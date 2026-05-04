from pathlib import Path
from dotenv import load_dotenv

# Load local .env (kept in src/.env) so optional API keys become available to the app.
load_dotenv(Path(__file__).resolve().parent / ".env")

from chatbot import chatbot


chatbot()
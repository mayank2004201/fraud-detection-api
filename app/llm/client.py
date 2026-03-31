from groq import Groq
from app.core.config import settings

def get_groq_client() -> Groq:
    """Return a fresh instance of the Groq client initialized with your API key."""
    if not settings.GROQ_API_KEY:
        # Fallback to direct environment check if Pydantic hasn't picked it up
        from os import getenv
        api_key = getenv("GROQ_API_KEY")
    else:
        api_key = settings.GROQ_API_KEY
        
    return Groq(api_key=api_key)

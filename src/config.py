import os
from dotenv import load_dotenv

# Load environment variables from .env (project root)
load_dotenv(override=True)

class Config:
    """
    Central configuration class.
    """

    # ========================
    # API KEYS
    # ========================
    # ========================
    # API KEYS
    # ========================
    # Try environment variable first, then Streamlit secrets
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    try:
        import streamlit as st
        if not GOOGLE_API_KEY and "GOOGLE_API_KEY" in st.secrets:
             GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
        if not OPENAI_API_KEY and "OPENAI_API_KEY" in st.secrets:
             OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    except (ImportError, FileNotFoundError, Exception):
        pass # Ignore if streamlit is not installed or secrets not found


    # ========================
    # GEMINI SETTINGS
    # ========================
    GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"
    OPENAI_MODEL_NAME = "gpt-3.5-turbo" # Or gpt-3.5-turbo if preferred

    # ========================
    # PATHS
    # ========================
    BASE_DIR = os.getcwd()
    DATA_DIR = os.path.join(BASE_DIR, "data")
    INPUT_DIR = os.path.join(DATA_DIR, "input")
    PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

    # ========================
    # VALIDATION
    # ========================
    @staticmethod
    def validate():
        if not Config.GOOGLE_API_KEY:
            raise ValueError(
                "❌ GOOGLE_API_KEY is missing. "
                "Please set it in .env (local) or Streamlit Secrets (cloud)."
            )
        if not Config.OPENAI_API_KEY or Config.OPENAI_API_KEY == "your_openai_api_key_here":
             raise ValueError(
                "❌ OPENAI_API_KEY is missing or is still the placeholder. "
                "Please replace 'your_openai_api_key_here' in .env with your real key."
            )

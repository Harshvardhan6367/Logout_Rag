import os
from dotenv import load_dotenv

# Load environment variables from .env (project root)
load_dotenv()

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
    try:
        import streamlit as st
        if not GOOGLE_API_KEY and "GOOGLE_API_KEY" in st.secrets:
             GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    except (ImportError, FileNotFoundError, Exception):
        pass # Ignore if streamlit is not installed or secrets not found


    # ========================
    # GEMINI SETTINGS
    # ========================
    GEMINI_MODEL_NAME = "gemini-2.5-flash-lite"

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

import os
import google.generativeai as genai
from dotenv import load_dotenv
import sys

# Load env vars
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    sys.path.append(os.getcwd())
    try:
        from src.config import Config
        api_key = Config.GOOGLE_API_KEY
    except:
        pass

if not api_key:
    exit(1)

genai.configure(api_key=api_key)

with open('models_clean.txt', 'w', encoding='utf-8') as f:
    try:
        found = False
        for m in genai.list_models():
            if 'embedContent' in m.supported_generation_methods:
                print(f"Model: {m.name}")
                f.write(f"{m.name}\n")
                found = True
        if not found:
            f.write("NO_MODELS_FOUND\n")
    except Exception as e:
        f.write(f"ERROR: {str(e)}\n")

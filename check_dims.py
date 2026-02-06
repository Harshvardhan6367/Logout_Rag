import json
import os
import google.generativeai as genai
from dotenv import load_dotenv
import numpy as np

# Load env vars
load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    # Try to extract from config as fallback
    try:
        import sys
        sys.path.append(os.getcwd())
        from src.config import Config
        api_key = Config.GOOGLE_API_KEY
    except:
        pass

if not api_key:
    print("No API Key found")
    exit(1)

genai.configure(api_key=api_key)

print("Checking current model dimensions for 'models/gemini-embedding-001':")
try:
    result = genai.embed_content(
        model="models/gemini-embedding-001",
        content="Hello world",
        task_type="retrieval_query"
    )
    # embed_content returns a dict with 'embedding' key which is a list
    embedding = result['embedding']
    print(f"Current model dimension: {len(embedding)}")
except Exception as e:
    print(f"Error embedding with current model: {e}")

def check_file(filename):
    path = os.path.join("data", "vectors", filename)
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                vectors = data.get("vectors", [])
                if vectors:
                    dim = len(vectors[0]["embedding"])
                    print(f"File '{filename}' has {len(vectors)} vectors with dimension: {dim}")
                else:
                    print(f"File '{filename}' is empty or has no vectors.")
        except Exception as e:
            print(f"Error reading '{filename}': {e}")
    else:
        print(f"File '{filename}' does not exist.")

print("\nChecking stored vectors:")
check_file("default.json")
check_file("otc_medicines.json")

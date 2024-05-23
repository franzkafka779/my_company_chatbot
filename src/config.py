import os
from dotenv import load_dotenv

load_dotenv()

# Example configuration loading
MODEL_NAME = os.getenv("MODEL_NAME", "llama3")
PERSIST_DIRECTORY = os.getenv("PERSIST_DIRECTORY", "db")

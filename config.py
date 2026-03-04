import os
from dotenv import load_dotenv
 
load_dotenv()
 
def must_get(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise ValueError(f"Missing environment variable: {name}")
    return v
 
# --- Azure OpenAI ---
AZURE_OPENAI_ENDPOINT = must_get("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = must_get("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = must_get("AZURE_OPENAI_API_VERSION")
AZURE_OPENAI_CHAT_DEPLOYMENT = must_get("AZURE_OPENAI_CHAT_DEPLOYMENT")
AZURE_OPENAI_EMBED_DEPLOYMENT = must_get("AZURE_OPENAI_EMBED_DEPLOYMENT")
 
# --- Azure AI Search ---
AZURE_SEARCH_ENDPOINT = must_get("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_KEY = must_get("AZURE_SEARCH_KEY")
AZURE_SEARCH_INDEX = must_get("AZURE_SEARCH_INDEX")
 
# --- Azure Blob ---
AZURE_BLOB_CONNECTION_STRING = must_get("AZURE_BLOB_CONNECTION_STRING")
AZURE_BLOB_CONTAINER = must_get("AZURE_BLOB_CONTAINER")
 
# --- Postgres ---
PG_HOST = must_get("PG_HOST")
PG_PORT = int(os.getenv("PG_PORT", "5432"))
PG_DATABASE = must_get("PG_DATABASE")
PG_USER = must_get("PG_USER")
PG_PASSWORD = must_get("PG_PASSWORD")
PG_SSLMODE = os.getenv("PG_SSLMODE", "require")
 
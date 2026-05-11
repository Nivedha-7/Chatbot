# Azure RAG Chatbot
A local Azure RAG chatbot using Streamlit, Azure OpenAI, Azure AI Search / Blob, and Postgres + pgvector.
 
## Prerequisites
- Python 3.10+ installed
- PostgreSQL with `pgvector` extension available
- Azure resources configured for OpenAI and Azure Search
- A `.env` file in the project root with required variables
 
## Install dependencies
Open PowerShell in `Chatbot-main` and run:
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

## Required environment variables
Create a `.env` file in `Chatbot-main` with these values:
```env
# Azure OpenAI
AZURE_OPENAI_ENDPOINT=
AZURE_OPENAI_API_KEY=
AZURE_OPENAI_API_VERSION=
AZURE_OPENAI_CHAT_DEPLOYMENT=
AZURE_OPENAI_EMBED_DEPLOYMENT=
 
# Azure AI Search
AZURE_SEARCH_ENDPOINT=
AZURE_SEARCH_KEY=
AZURE_SEARCH_INDEX=
EMBEDDING_DIM=1536
 
# Azure Blob storage (used by some scripts)
AZURE_BLOB_CONNECTION_STRING=
AZURE_BLOB_CONTAINER=
 
# Postgres
PG_HOST=
PG_PORT=5432
PG_DATABASE=
PG_USER=
PG_PASSWORD=
PG_SSLMODE=require
```
> Note: `EMBEDDING_DIM` should match the embedding model dimension used by Azure OpenAI.
## Initialize Postgres / pgvector schema
Run the database setup script to create the required tables and extension:
python create_index.py

## Validate connections
Test the database connection:
python test_db.py

Test Azure OpenAI embedding access:
python test_azure_openai.py

## Run the app
Start the Streamlit app:
streamlit run app.py

Then open the URL shown in the terminal (usually `http://localhost:8501`).
## Optional commands
- `python quick_embed_test.py` — quick test for embedding generation
- `python init_db.py` — alternative DB init helper if available
- `python insert_test.py` — test insert logic against Postgres
- `python app_langchain.py` — if you want to inspect the alternative LangChain app file
 

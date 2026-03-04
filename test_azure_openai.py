import os
from dotenv import load_dotenv
from openai import AzureOpenAI
 
load_dotenv()
 
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)
 
DEPLOY = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
 
print("Endpoint:", os.getenv("AZURE_OPENAI_ENDPOINT"))
print("Embed deployment:", DEPLOY)
 
resp = client.embeddings.create(
    model=DEPLOY,
    input="hello world"
)
 
print("✅ Embedding length:", len(resp.data[0].embedding))
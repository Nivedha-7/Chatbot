import os
from dotenv import load_dotenv
from openai import AzureOpenAI
 
load_dotenv(override=True)
 
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
)
 
dep = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
print("ENDPOINT:", os.getenv("AZURE_OPENAI_ENDPOINT"))
print("EMBED DEP:", dep)
 
r = client.embeddings.create(
    model=dep,
    input=["test embedding"]
)
print("OK vector length:", len(r.data[0].embedding))
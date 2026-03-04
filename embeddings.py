# embeddings.py
import os
from typing import List
 
from dotenv import load_dotenv
from openai import AzureOpenAI
 
load_dotenv(override=True)
 
def _get_client() -> AzureOpenAI:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION")
 
    missing = [k for k, v in {
        "AZURE_OPENAI_ENDPOINT": endpoint,
        "AZURE_OPENAI_API_KEY": api_key,
        "AZURE_OPENAI_API_VERSION": api_version,
    }.items() if not v]
 
    if missing:
        raise RuntimeError(f"Missing env values: {', '.join(missing)}")
 
    return AzureOpenAI(
        api_key=api_key,
        api_version=api_version,
        azure_endpoint=endpoint,
    )
 
def embed_texts(texts: List[str]) -> List[List[float]]:
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        raise ValueError("embed_texts expects a list[str]")
 
    deployment = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")
    if not deployment:
        raise RuntimeError("Missing env value: AZURE_OPENAI_EMBED_DEPLOYMENT")
 
    clean_texts = [(t or "").strip() for t in texts]
    if not any(clean_texts):
        return []
 
    client = _get_client()
 
    resp = client.embeddings.create(
        model=deployment,      # Azure uses deployment name here
        input=clean_texts
    )
 
    return [item.embedding for item in resp.data]
 
 
if __name__ == "__main__":
    vecs = embed_texts(["hello world", "this is a test"])
    print("Got vectors:", len(vecs))
    print("Vector length:", len(vecs[0]))
 
from typing import List, Dict, Any, Optional
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential
 
from config import AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_KEY, AZURE_SEARCH_INDEX
 
# Create client once
_search_client = SearchClient(
    endpoint=AZURE_SEARCH_ENDPOINT,
    index_name=AZURE_SEARCH_INDEX,
    credential=AzureKeyCredential(AZURE_SEARCH_KEY)
)
 
def upload_chunks(docs_for_search: List[Dict[str, Any]]) -> None:
    """
    Upload chunks to Azure AI Search.
    docs_for_search must contain keys:
      id, doc_id, filename, chunk_index, content, contentVector
    """
    if not docs_for_search:
        return
 
    result = _search_client.upload_documents(documents=docs_for_search)
 
    # If any failed, raise error with details
    failed = [r for r in result if not r.succeeded]
    if failed:
        msgs = []
        for f in failed[:5]:
            msgs.append(f"key={f.key} error={getattr(f, 'error_message', 'unknown')}")
        raise RuntimeError("Some documents failed to upload: " + " | ".join(msgs))
 
def vector_search(query_vector: List[float], top_k: int = 5, doc_id: Optional[str] = None):
    """
    Vector search on contentVector field.
    """
    vector_query = VectorizedQuery(
        vector=query_vector,
        k_nearest_neighbors=top_k,
        fields="contentVector"
    )
 
    filter_expr = None
    if doc_id:
        filter_expr = f"doc_id eq '{doc_id}'"
 
    results = _search_client.search(
        search_text="*",  # required; vector does the real work
        vector_queries=[vector_query],
        select=["id", "doc_id", "filename", "chunk_index", "content"],
        filter=filter_expr,
        top=top_k
    )
 
    out = []
    for r in results:
        out.append({
            "id": r.get("id"),
            "doc_id": r.get("doc_id"),
            "filename": r.get("filename"),
            "chunk_index": r.get("chunk_index"),
            "content": r.get("content"),
            "score": r.get("@search.score"),
        })
    return out
 
def list_documents(limit: int = 50):
    """
    Returns a list of unique documents seen in the index.
    """
    results = _search_client.search(
        search_text="*",
        select=["doc_id", "filename"],
        top=limit
    )
 
    seen = set()
    docs = []
    for r in results:
        d = (r.get("doc_id"), r.get("filename"))
        if d not in seen and d[0] and d[1]:
            seen.add(d)
            docs.append({"doc_id": d[0], "filename": d[1]})
    return docs
 
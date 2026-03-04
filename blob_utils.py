from azure.storage.blob import BlobServiceClient, ContentSettings
from config import AZURE_BLOB_CONNECTION_STRING, AZURE_BLOB_CONTAINER
import uuid
 
_bsc = BlobServiceClient.from_connection_string(AZURE_BLOB_CONNECTION_STRING)
_container = _bsc.get_container_client(AZURE_BLOB_CONTAINER)
 
def ensure_container():
    try:
        _container.create_container()
    except Exception:
        pass
 
def upload_file_bytes(filename: str, data: bytes) -> tuple[str, str]:
    ensure_container()
    blob_name = f"{uuid.uuid4().hex}_{filename}"
    blob = _container.get_blob_client(blob_name)
    blob.upload_blob(
        data,
        overwrite=True,
        content_settings=ContentSettings(content_type="application/octet-stream")
    )
    return blob_name, blob.url

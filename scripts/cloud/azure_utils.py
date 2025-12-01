"""
Azure Blob Storage utilities for downloading audio/video files.
Reuses authentication logic from loc_vhp_2_az_blob.py
"""
import os
import io
import logging
from typing import Optional, BinaryIO
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential

# Suppress Azure SDK verbose logging
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)


def make_blob_service() -> BlobServiceClient:
    """
    Create authenticated Azure Blob Service client.
    Reads configuration from environment variables:
    - AZURE_STORAGE_CONTAINER: blob container name
    - AZURE_AUTH: "connection_string" or "managed_identity" (default)
    - AZURE_STORAGE_CONNECTION_STRING: required if using connection_string
    - AZURE_STORAGE_ACCOUNT: required if using managed_identity
    """
    container = os.environ["AZURE_STORAGE_CONTAINER"]
    auth_mode = os.environ.get("AZURE_AUTH", "managed_identity")

    if auth_mode == "connection_string":
        conn = os.environ["AZURE_STORAGE_CONNECTION_STRING"]
        svc = BlobServiceClient.from_connection_string(conn)
    else:
        # managed identity / CLI / env creds
        acct = os.environ["AZURE_STORAGE_ACCOUNT"]
        cred = DefaultAzureCredential(exclude_interactive_browser_credential=False)
        svc = BlobServiceClient(f"https://{acct}.blob.core.windows.net", credential=cred)

    return svc


def download_blob_to_memory(blob_path: str, container: Optional[str] = None) -> bytes:
    """
    Download a blob from Azure storage to memory (returns bytes).

    Args:
        blob_path: Path to blob (e.g., "vhp/0/video.mp4")
        container: Container name (defaults to env AZURE_STORAGE_CONTAINER)

    Returns:
        File contents as bytes
    """
    svc = make_blob_service()
    container = container or os.environ["AZURE_STORAGE_CONTAINER"]

    blob_client = svc.get_blob_client(container=container, blob=blob_path)

    # Download to BytesIO
    stream = io.BytesIO()
    blob_client.download_blob().readinto(stream)
    stream.seek(0)

    return stream.read()


def download_blob_to_stream(blob_path: str, container: Optional[str] = None) -> BinaryIO:
    """
    Download a blob from Azure storage and return as a stream (BytesIO).
    Useful for passing directly to audio processing libraries.

    Args:
        blob_path: Path to blob (e.g., "vhp/0/video.mp4")
        container: Container name (defaults to env AZURE_STORAGE_CONTAINER)

    Returns:
        BytesIO stream positioned at the start
    """
    svc = make_blob_service()
    container = container or os.environ["AZURE_STORAGE_CONTAINER"]

    blob_client = svc.get_blob_client(container=container, blob=blob_path)

    # Download to BytesIO
    stream = io.BytesIO()
    blob_client.download_blob().readinto(stream)
    stream.seek(0)

    return stream


def blob_exists(blob_path: str, container: Optional[str] = None) -> bool:
    """
    Check if a blob exists in Azure storage.

    Args:
        blob_path: Path to blob
        container: Container name (defaults to env AZURE_STORAGE_CONTAINER)

    Returns:
        True if blob exists, False otherwise
    """
    try:
        svc = make_blob_service()
        container = container or os.environ["AZURE_STORAGE_CONTAINER"]
        blob_client = svc.get_blob_client(container=container, blob=blob_path)
        blob_client.get_blob_properties()
        return True
    except Exception:
        return False


def upload_blob(blob_path: str, data: bytes, container: Optional[str] = None, overwrite: bool = True) -> None:
    """
    Upload bytes data to Azure blob storage.

    Args:
        blob_path: Path to blob (e.g., "loc_vhp/123/123_000.wav")
        data: File contents as bytes
        container: Container name (defaults to env AZURE_STORAGE_CONTAINER)
        overwrite: Whether to overwrite existing blob (default True)
    """
    svc = make_blob_service()
    container = container or os.environ["AZURE_STORAGE_CONTAINER"]

    blob_client = svc.get_blob_client(container=container, blob=blob_path)
    blob_client.upload_blob(data, overwrite=overwrite)


def list_blobs(prefix: str, container: Optional[str] = None) -> list[str]:
    """
    List all blobs with a given prefix.

    Args:
        prefix: Blob path prefix (e.g., "vhp/")
        container: Container name (defaults to env AZURE_STORAGE_CONTAINER)

    Returns:
        List of blob paths
    """
    svc = make_blob_service()
    container = container or os.environ["AZURE_STORAGE_CONTAINER"]

    container_client = svc.get_container_client(container)
    blobs = container_client.list_blobs(name_starts_with=prefix)

    return [blob.name for blob in blobs]

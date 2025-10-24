import os, time, hashlib, mimetypes
from urllib.parse import urlparse
from typing import Optional
import requests
import pandas as pd
from tqdm import tqdm

from azure.storage.blob import BlobServiceClient, ContentSettings
from azure.identity import DefaultAzureCredential

# ---------- Azure client factory ----------
def make_blob_service() -> BlobServiceClient:
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
    # Ensure container exists (no-op if it already does)
    try:
        svc.create_container(container)
    except Exception:
        pass
    return svc

# ---------- small utils ----------
def _ext_from_url_or_ct(url: str, content_type: Optional[str]) -> str:
    # 1) try server content-type
    if content_type:
        if "mpeg" in content_type: return ".mp3"
        if "mp4" in content_type: return ".mp4"
        if "wav" in content_type: return ".wav"
        if "x-m4a" in content_type or "aac" in content_type: return ".m4a"
    # 2) try URL suffix
    path = urlparse(url).path.lower()
    for ext in [".mp3", ".mp4", ".wav", ".m4a", ".flac", ".aac"]:
        if path.endswith(ext): return ext
    # 3) fallback
    guess = mimetypes.guess_extension(content_type or "")
    return guess or ".bin"

def _blob_exists(blob_client) -> bool:
    try:
        blob_client.get_blob_properties()
        return True
    except Exception:
        return False

# ---------- main downloader/uploader ----------
def upload_media_to_blob(df: pd.DataFrame,
                         prefix: str = "vhp",
                         max_bytes: int = 2_000_000_000,
                         timeout: int = 60):
    """
    For each row, prefer video_url else audio_url, stream-download,
    and upload to Azure Blob at: {prefix}/{idx}/{media_type}{ext}
    Adds metadata: media_type, source_url, sn_bytes, ts_epoch.
    """
    svc = make_blob_service()
    container = os.environ["AZURE_STORAGE_CONTAINER"]
    session = requests.Session()
    session.headers.update({"User-Agent": "amia-download/1.0"})

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="uploading"):
        media_url = None
        media_type = None
        if pd.notnull(row.get("video_url")) and str(row["video_url"]).strip():
            media_type, media_url = "video", str(row["video_url"]).strip()
        elif pd.notnull(row.get("audio_url")) and str(row["audio_url"]).strip():
            media_type, media_url = "audio", str(row["audio_url"]).strip()
        else:
            print(f"[{idx}] no media_url; skipping")
            continue

        # HEAD first (size/type)
        try:
            head = session.head(media_url, allow_redirects=True, timeout=timeout)
        except Exception as e:
            print(f"[{idx}] HEAD failed: {e}; trying GET anyway…")
            head = None

        content_len = None
        content_type = None
        if head is not None and head.ok:
            content_type = head.headers.get("Content-Type")
            try:
                content_len = int(head.headers.get("Content-Length", "0"))
            except Exception:
                content_len = None

        ext = _ext_from_url_or_ct(media_url, content_type)
        blob_path = f"{prefix}/{idx}/{media_type}{ext}"
        blob = svc.get_blob_client(container=container, blob=blob_path)

        # skip if exists
        if _blob_exists(blob):
            print(f"[{idx}] exists: {blob_path} (skip)")
            continue

        # size guard
        if content_len and content_len > max_bytes:
            print(f"[{idx}] too large ({content_len} bytes): {media_url}")
            continue

        # GET stream + upload stream (chunks)
        try:
            with session.get(media_url, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                ctype = resp.headers.get("Content-Type") or content_type
                ext = _ext_from_url_or_ct(media_url, ctype)

                def gen():
                    md5 = hashlib.md5()
                    total = 0
                    for chunk in resp.iter_content(chunk_size=4 * 1024 * 1024):
                        if not chunk:
                            continue
                        md5.update(chunk)
                        total += len(chunk)
                        yield chunk
                        if total > max_bytes:
                            raise RuntimeError("File exceeds max_bytes limit")
                # upload stream
                blob.upload_blob(
                    data=gen(),
                    overwrite=False,
                    max_concurrency=4,
                    timeout=600,
                    content_settings=ContentSettings(content_type=ctype or "application/octet-stream"),
                )
                # add metadata
                blob.set_blob_metadata({
                    "media_type": media_type,
                    "source_url": media_url[:2000],
                    "ts_epoch": str(int(time.time())),
                })
                print(f"[{idx}] uploaded: {blob_path}")
        except Exception as e:
            print(f"[{idx}] FAILED {media_type} → {media_url}: {e}")

# ---- example usage ----
if __name__ == "__main__":
    df = pd.read_parquet("data/raw/loc/veterans_history_project_resources.parquet")
    upload_media_to_blob(df, prefix="unknown")
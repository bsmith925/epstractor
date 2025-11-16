# src/download.py

from __future__ import annotations

import concurrent.futures
import hashlib
import io
import json
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple
from urllib.parse import urlparse, unquote

import requests
import yaml
from dotenv import load_dotenv
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import credentials as oauth_credentials
from google.oauth2 import service_account
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaIoBaseDownload
from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    MofNCompleteColumn,
)

load_dotenv()

logger = logging.getLogger(__name__)

ItemKind = Literal["http_file", "gdrive_folder"]

# Delay between downloads to avoid rate limiting
FILE_DOWNLOAD_DELAY = 2.0  # seconds between file downloads

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


@dataclass
class DownloadItem:
    """A single item to download."""

    kind: ItemKind
    url: Optional[str] = None
    filename: Optional[str] = None
    folder_id: Optional[str] = None
    recursive: bool = False


@dataclass
class SourceConfig:
    """Parsed configuration for a single source YAML."""

    source: str
    description: Optional[str]
    output_dir: Path
    subdir: str
    items: List[DownloadItem]


class Downloader:
    """
    Downloads files defined in a YAML config.

    Supports:
    - HTTP file downloads
    - Google Drive folder downloads (using Google Drive API)

    Authentication methods (in order of preference):
    1. Service account (most reliable, requires sharing folder with service account email)
    2. OAuth2 credentials (works for public folders, requires credentials.json)
    3. API key (may have issues with automated query blocking, but works for public folders)
    """

    def __init__(
        self,
        config_path: Path,
        base_output_dir: Optional[Path] = None,
        api_key: Optional[str] = None,
        service_account_file: Optional[str] = None,
        credentials_file: Optional[str] = None,
        max_http_workers: int = 8,
        max_drive_workers: int = 4,
        manifest_only: bool = False,
        verify_only: bool = False,
        verify_downloads: bool = True,
        use_progress: bool = True,
        console: Optional[Console] = None,
    ) -> None:
        self.config_path = Path(config_path)
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self.service_account_file = (
            service_account_file
            or os.environ.get("GOOGLE_SERVICE_ACCOUNT_FILE")
            or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
        )
        self.credentials_file = credentials_file or os.environ.get(
            "GOOGLE_CREDENTIALS_FILE"
        )
        if not self.credentials_file:
            # Prefer credentials.json; fall back to credentials_saved.json if present
            if Path("credentials.json").exists():
                self.credentials_file = "credentials.json"
            elif Path("credentials_saved.json").exists():
                self.credentials_file = "credentials_saved.json"
            else:
                self.credentials_file = "credentials.json"
        self.config = self._load_config(base_output_dir)
        if self.api_key:
            logger.warning(
                "API key authentication is not supported by the Google Drive API client. "
                "Provide OAuth2 credentials or a service account instead."
            )
        self.session = requests.Session()
        self.max_http_workers = max(1, max_http_workers)
        self.max_drive_workers = max(1, max_drive_workers)
        self.manifest_only = manifest_only
        self.verify_only = verify_only
        self.verify_downloads = verify_downloads
        self._session_local: threading.local = threading.local()
        self.use_progress = use_progress
        self.console: Optional[Console] = console if use_progress else None
        credentials_path = Path(self.credentials_file)
        token_override = os.environ.get("GOOGLE_DRIVE_TOKEN_FILE")
        if token_override:
            self._token_file = Path(token_override)
        else:
            self._token_file = (
                credentials_path.parent / f"{credentials_path.stem}.token.json"
            )
        self._drive_credentials = None
        self._drive_local: threading.local = threading.local()

    def _load_config(self, base_output_dir: Optional[Path]) -> SourceConfig:
        """Load and parse YAML config."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with self.config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        if not isinstance(data, dict):
            raise ValueError(f"Config must be a mapping at root: {self.config_path}")

        source = data.get("source")
        if not source:
            raise ValueError("Config is missing required key: 'source'")

        # Determine output directory
        yaml_output_dir = data.get("output_dir", "downloads/datasets")
        if base_output_dir is not None:
            output_root = Path(base_output_dir)
        else:
            output_root = Path(yaml_output_dir)

        subdir = data.get("subdir", source)

        # Parse items
        items_raw = data.get("items", [])
        if not items_raw:
            raise ValueError("Config must contain a non-empty 'items' list")

        items: List[DownloadItem] = []
        for item in items_raw:
            if not isinstance(item, dict):
                raise ValueError("Each item in 'items' must be a mapping")

            kind: ItemKind = item.get("kind", "http_file")

            if kind == "http_file":
                url = item.get("url")
                if not url:
                    raise ValueError("http_file item missing required 'url'")
                items.append(
                    DownloadItem(
                        kind="http_file",
                        url=url,
                        filename=item.get("filename"),
                    )
                )
            elif kind == "gdrive_folder":
                folder_id = item.get("folder_id")
                if not folder_id:
                    raise ValueError("gdrive_folder item missing 'folder_id'")
                items.append(
                    DownloadItem(
                        kind="gdrive_folder",
                        folder_id=folder_id,
                        recursive=bool(item.get("recursive", False)),
                    )
                )
            else:
                raise ValueError(f"Unknown item kind: {kind}")

        return SourceConfig(
            source=source,
            description=data.get("description"),
            output_dir=output_root,
            subdir=subdir,
            items=items,
        )

    def _get_drive_credentials(self):
        """Load Google Drive credentials using service account or OAuth2."""
        if self._drive_credentials is not None:
            return self._drive_credentials

        # Service account takes precedence when configured
        if self.service_account_file:
            service_account_path = Path(self.service_account_file)
            if not service_account_path.exists():
                raise RuntimeError(
                    f"Service account file not found: {service_account_path}\n"
                    "Ensure the path is correct and the file exists."
                )

            logger.info("Using service account authentication")
            credentials = service_account.Credentials.from_service_account_file(
                str(service_account_path), scopes=DRIVE_SCOPES
            )
            self._drive_credentials = credentials
            return credentials

        credentials_path = Path(self.credentials_file)
        if not credentials_path.exists():
            raise RuntimeError(
                "OAuth2 credentials file not found. Provide credentials.json via:\n"
                "  --credentials-file /path/to/credentials.json\n"
                "  or set GOOGLE_CREDENTIALS_FILE=/path/to/credentials.json"
            )

        creds = None
        if self._token_file.exists():
            try:
                creds = oauth_credentials.Credentials.from_authorized_user_file(
                    str(self._token_file), scopes=DRIVE_SCOPES
                )
            except Exception as exc:
                logger.warning(
                    "Failed to load saved token %s: %s", self._token_file, exc
                )

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                logger.info("Refreshing OAuth2 credentials")
                creds.refresh(GoogleAuthRequest())
            else:
                logger.info("Starting OAuth2 flow...")
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(credentials_path), scopes=DRIVE_SCOPES
                )
                try:
                    # In tmux/headless environments, avoid trying to open a browser
                    creds = flow.run_local_server(port=0, open_browser=False)
                except Exception as exc:
                    logger.warning(
                        "Local server auth failed (%s). Falling back to console auth. "
                        "Follow the printed URL and paste the code.",
                        exc,
                    )
                    creds = flow.run_console()

            try:
                self._token_file.write_text(creds.to_json(), encoding="utf-8")
                logger.info("Saved OAuth2 token to %s", self._token_file)
            except Exception as exc:
                logger.warning(
                    "Failed to save OAuth2 token %s: %s", self._token_file, exc
                )

        self._drive_credentials = creds
        return creds

    def _get_drive_service(self):
        """Return a thread-local Google Drive API client."""
        service = getattr(self._drive_local, "service", None)
        if service is not None:
            return service

        credentials = self._get_drive_credentials()
        service = build("drive", "v3", credentials=credentials, cache_discovery=False)
        self._drive_local.service = service
        return service

    def download_all(self, overwrite: bool = False) -> None:
        """Download all items defined in the config."""
        target_root = self.config.output_dir / self.config.subdir
        target_root.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Downloading %d item(s) for source '%s' into: %s",
            len(self.config.items),
            self.config.source,
            target_root,
        )

        http_items = [item for item in self.config.items if item.kind == "http_file"]
        gdrive_items = [
            item for item in self.config.items if item.kind == "gdrive_folder"
        ]

        if http_items:
            self._download_http_files(http_items, target_root, overwrite)

        for item in gdrive_items:
            self._download_gdrive_folder(item, target_root, overwrite)

    def _get_http_session(self) -> requests.Session:
        session = getattr(self._session_local, "session", None)
        if session is None:
            session = requests.Session()
            self._session_local.session = session
        return session

    def _download_http_files(
        self, items: List[DownloadItem], target_root: Path, overwrite: bool
    ) -> None:
        if self.manifest_only:
            logger.info("Manifest-only mode: skipping HTTP downloads.")
            return

        if self.verify_only:
            logger.info(
                "Verify-only mode: skipping HTTP downloads (no manifest available)."
            )
            return

        logger.info(
            "Downloading %d HTTP file(s) with up to %d workers",
            len(items),
            self.max_http_workers,
        )

        def worker(item: DownloadItem) -> None:
            self._download_http_file(item, target_root, overwrite)

        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_http_workers
        )
        futures = [executor.submit(worker, item) for item in items]

        if self.use_progress and len(futures) > 0:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold]HTTP[/]"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
                refresh_per_second=8,
            ) as progress:
                task_id = progress.add_task("http_files", total=len(futures))
                for future in concurrent.futures.as_completed(futures):
                    future.result()
                    progress.advance(task_id, 1)
        else:
            for future in concurrent.futures.as_completed(futures):
                future.result()

        executor.shutdown(wait=True)

    def _download_http_file(
        self, item: DownloadItem, target_root: Path, overwrite: bool
    ) -> None:
        """Download a file via HTTP."""
        if not item.url:
            raise ValueError("http_file item missing url")

        # Determine filename
        if item.filename:
            filename = item.filename
        else:
            parsed = urlparse(item.url)
            filename = Path(unquote(parsed.path)).name
            if not filename:
                raise ValueError(f"Cannot infer filename from URL: {item.url}")

        dest_path = target_root / filename

        if dest_path.exists() and not overwrite:
            logger.debug("Skipping existing file: %s", dest_path)
            return

        logger.info("Downloading %s -> %s", item.url, dest_path)

        session = self._get_http_session()

        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            with session.get(item.url, stream=True, timeout=60) as resp:
                resp.raise_for_status()

                tmp_path = dest_path.with_suffix(dest_path.suffix + ".part")
                downloaded = 0
                with tmp_path.open("wb") as f:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                tmp_path.rename(dest_path)
                size_str = (
                    f"{downloaded / 1_048_576:.2f} MB" if downloaded else "unknown size"
                )
                logger.info("Saved %s (%s)", dest_path, size_str)
        except Exception as e:
            logger.error("Failed to download %s: %s", item.url, e)
            raise

    def _download_gdrive_folder(
        self, item: DownloadItem, target_root: Path, overwrite: bool
    ) -> None:
        """Download files from a Google Drive folder using the Drive API."""
        if not item.folder_id:
            raise ValueError("gdrive_folder item missing folder_id")

        drive_service = self._get_drive_service()

        logger.info(
            "Downloading Google Drive folder %s (recursive=%s) using Google Drive API",
            item.folder_id,
            item.recursive,
        )

        # Stream listing and downloads concurrently; write manifest at the end
        manifest: List[Dict[str, Any]] = []
        futures: List[concurrent.futures.Future] = []
        scheduled_count = 0
        # Dedupe controls: skip scheduling identical path+md5; track conflicts where same path has different md5
        seen_md5_by_path: Dict[str, set] = {}
        skipped_duplicate_count = 0
        conflict_md5s_by_path: Dict[str, set] = {}

        executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_drive_workers
        )

        def submit_download(file_meta: Dict[str, Any], dest_path: Path) -> None:
            nonlocal scheduled_count
            future = executor.submit(
                self._download_gdrive_file,
                file_meta["id"],
                file_meta.get("name") or file_meta.get("title") or file_meta["id"],
                dest_path,
                file_meta.get("md5Checksum"),
                file_meta.get("size"),
            )
            futures.append(future)
            scheduled_count += 1
            if self.use_progress and downloads_task is not None:
                progress.update(downloads_task, total=scheduled_count)

        generator = self._walk_folder(
            drive_service, item.folder_id, target_root, item.recursive
        )

        if self.use_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold]Listing[/]"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                SpinnerColumn(),
                TextColumn("[bold]Downloads[/]"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
                refresh_per_second=8,
            ) as progress:
                listing_task = progress.add_task("drive_listing", total=None)
                downloads_task = (
                    progress.add_task("drive_files", total=0)
                    if not self.manifest_only and not self.verify_only
                    else None
                )

                for file_meta, dest_path in generator:
                    # Build manifest entry
                    manifest.append(
                        {
                            "id": file_meta["id"],
                            "path": str(dest_path.relative_to(target_root)),
                            "md5": file_meta.get("md5Checksum"),
                            "size": file_meta.get("size"),
                        }
                    )
                    progress.advance(listing_task, 1)

                    if self.manifest_only or self.verify_only:
                        continue

                    # Decide whether to schedule
                    expected_size = file_meta.get("size")
                    expected_md5 = file_meta.get("md5Checksum")
                    rel_path_str = str(dest_path.relative_to(target_root))
                    if expected_md5:
                        existing = seen_md5_by_path.get(rel_path_str)
                        if existing is None:
                            seen_md5_by_path[rel_path_str] = {expected_md5}
                        else:
                            if expected_md5 in existing:
                                skipped_duplicate_count += 1
                                # Identical path+md5 duplicate: skip scheduling
                                continue
                            else:
                                existing.add(expected_md5)
                                conflict_md5s_by_path[rel_path_str] = set(existing)
                    if dest_path.exists() and not overwrite:
                        if self.verify_downloads and self._verify_local_file(
                            dest_path, expected_size, expected_md5
                        ):
                            continue
                        elif not self.verify_downloads:
                            continue

                    submit_download(file_meta, dest_path)

                # After listing completes
                if skipped_duplicate_count:
                    logger.info(
                        "Deduped %d download(s) by identical path+md5",
                        skipped_duplicate_count,
                    )
                if conflict_md5s_by_path:
                    sample = list(conflict_md5s_by_path.keys())[:10]
                    logger.warning(
                        "Detected %d path(s) with multiple different md5 values (possible naming conflicts). Example(s): %s%s",
                        len(conflict_md5s_by_path),
                        ", ".join(sample),
                        "..." if len(conflict_md5s_by_path) > len(sample) else "",
                    )
                self._write_manifest(target_root, manifest)
                if self.manifest_only:
                    executor.shutdown(wait=False)
                    return
                if self.verify_only:
                    self._verify_against_manifest(target_root, manifest)
                    executor.shutdown(wait=False)
                    return

                # Wait for all downloads to complete
                for future in concurrent.futures.as_completed(futures):
                    future.result()
                    if downloads_task is not None:
                        progress.advance(downloads_task, 1)
        else:
            last_log = time.time()
            for file_meta, dest_path in generator:
                manifest.append(
                    {
                        "id": file_meta["id"],
                        "path": str(dest_path.relative_to(target_root)),
                        "md5": file_meta.get("md5Checksum"),
                        "size": file_meta.get("size"),
                    }
                )
                if time.time() - last_log >= 5:
                    logger.info("Enumerated %d file(s) so far...", len(manifest))
                    last_log = time.time()

                if not self.manifest_only and not self.verify_only:
                    expected_size = file_meta.get("size")
                    expected_md5 = file_meta.get("md5Checksum")
                    rel_path_str = str(dest_path.relative_to(target_root))
                    if expected_md5:
                        existing = seen_md5_by_path.get(rel_path_str)
                        if existing is None:
                            seen_md5_by_path[rel_path_str] = {expected_md5}
                        else:
                            if expected_md5 in existing:
                                skipped_duplicate_count += 1
                                # Identical path+md5 duplicate: skip scheduling
                                continue
                            else:
                                existing.add(expected_md5)
                                conflict_md5s_by_path[rel_path_str] = set(existing)
                    if dest_path.exists() and not overwrite:
                        if self.verify_downloads and self._verify_local_file(
                            dest_path, expected_size, expected_md5
                        ):
                            continue
                        elif not self.verify_downloads:
                            continue
                    submit_download(file_meta, dest_path)

            self._write_manifest(target_root, manifest)
            if skipped_duplicate_count:
                logger.info(
                    "Deduped %d download(s) by identical path+md5",
                    skipped_duplicate_count,
                )
            if conflict_md5s_by_path:
                sample = list(conflict_md5s_by_path.keys())[:10]
                logger.warning(
                    "Detected %d path(s) with multiple different md5 values (possible naming conflicts). Example(s): %s%s",
                    len(conflict_md5s_by_path),
                    ", ".join(sample),
                    "..." if len(conflict_md5s_by_path) > len(sample) else "",
                )
            if self.manifest_only:
                executor.shutdown(wait=False)
                return
            if self.verify_only:
                self._verify_against_manifest(target_root, manifest)
                executor.shutdown(wait=False)
                return

            for future in concurrent.futures.as_completed(futures):
                future.result()

        executor.shutdown(wait=True)

    def _walk_folder(
        self,
        drive_service: Any,
        folder_id: str,
        local_root: Path,
        recursive: bool = True,
        page_size: int = 1000,
    ) -> Iterable[Tuple[Dict[str, Any], Path]]:
        """
        Walk a Google Drive folder recursively using the Drive API,
        yielding (file_metadata, local_path) tuples.
        """
        stack = [(folder_id, local_root)]

        while stack:
            current_folder_id, current_local = stack.pop()
            page_token: Optional[str] = None

            while True:
                query = f"'{current_folder_id}' in parents and trashed = false"
                try:
                    response = (
                        drive_service.files()
                        .list(
                            q=query,
                            spaces="drive",
                            fields="nextPageToken, files(id,name,mimeType,md5Checksum,size)",
                            pageToken=page_token,
                            pageSize=page_size,
                            supportsAllDrives=True,
                            includeItemsFromAllDrives=True,
                        )
                        .execute()
                    )
                except HttpError as exc:
                    logger.error("Failed to list folder %s: %s", current_folder_id, exc)
                    raise

                for file_item in response.get("files", []):
                    mime_type = file_item.get("mimeType")
                    name = file_item.get("name") or file_item["id"]

                    if mime_type == "application/vnd.google-apps.folder" and recursive:
                        next_local = current_local / name
                        stack.append((file_item["id"], next_local))
                    elif mime_type != "application/vnd.google-apps.folder":
                        metadata = {
                            "id": file_item["id"],
                            "name": name,
                            "mimeType": mime_type,
                            "md5Checksum": file_item.get("md5Checksum"),
                            "size": int(file_item["size"])
                            if file_item.get("size")
                            else None,
                        }
                        yield metadata, current_local / name

                page_token = response.get("nextPageToken")
                if not page_token:
                    break

    def _download_gdrive_file(
        self,
        file_id: str,
        name: str,
        dest_path: Path,
        expected_md5: Optional[str],
        expected_size: Optional[int],
    ) -> None:
        """Download a single file from Google Drive using the Drive API."""
        logger.info("Downloading Drive file %s (%s) -> %s", file_id, name, dest_path)
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        # Use a unique temp file per Drive file to avoid races when multiple entries map to the same destination
        tmp_path = dest_path.with_suffix(dest_path.suffix + f".part.{file_id}")

        try:
            drive_service = self._get_drive_service()
            request = drive_service.files().get_media(
                fileId=file_id, supportsAllDrives=True
            )

            with io.FileIO(tmp_path, "wb") as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    try:
                        status, done = downloader.next_chunk()
                        if status:
                            logger.debug(
                                "Drive download %s: %.2f%% complete",
                                name,
                                status.progress() * 100,
                            )
                    except HttpError as exc:
                        logger.error(
                            "HTTP error while downloading %s (%s): %s",
                            name,
                            file_id,
                            exc,
                        )
                        raise

            tmp_path.replace(dest_path)

            if self.verify_downloads and not self._verify_local_file(
                dest_path, expected_size, expected_md5
            ):
                raise RuntimeError(
                    f"Verification failed for {dest_path} (expected size={expected_size}, md5={expected_md5})"
                )

            logger.info("Saved %s", dest_path)
        except Exception as e:
            logger.error("Failed to download %s: %s", name, e)
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            if dest_path.exists():
                dest_path.unlink()
            raise

    def _write_manifest(
        self, target_root: Path, manifest: List[Dict[str, Any]]
    ) -> None:
        if not manifest:
            return

        manifest_path = target_root / ".manifest.json"
        try:
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            logger.info(
                "Wrote manifest with %d entries to %s", len(manifest), manifest_path
            )
        except Exception as exc:
            logger.warning("Failed to write manifest %s: %s", manifest_path, exc)

    def _verify_against_manifest(
        self, target_root: Path, manifest: List[Dict[str, Any]]
    ) -> None:
        missing = []
        for entry in manifest:
            expected_path = target_root / entry["path"]
            if not self._verify_local_file(
                expected_path,
                int(entry["size"]) if entry.get("size") else None,
                entry.get("md5"),
            ):
                missing.append(entry["path"])
        if missing:
            logger.error(
                "Verification failed. %d file(s) missing or corrupt: %s",
                len(missing),
                ", ".join(missing[:10]) + ("..." if len(missing) > 10 else ""),
            )
            raise RuntimeError("Verification failed - see log for details")

        logger.info(
            "Verification succeeded for %d file(s) in %s",
            len(manifest),
            target_root,
        )

    def _verify_local_file(
        self,
        path: Path,
        expected_size: Optional[int],
        expected_md5: Optional[str],
    ) -> bool:
        if not path.exists():
            return False

        if expected_size is not None and path.stat().st_size != expected_size:
            logger.warning("Size mismatch for %s", path)
            return False

        if self.verify_downloads and expected_md5:
            return self._calculate_md5(path) == expected_md5

        return True

    @staticmethod
    def _calculate_md5(path: Path, chunk_size: int = 1_048_576) -> str:
        digest = hashlib.md5()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(chunk_size), b""):
                digest.update(chunk)
        return digest.hexdigest()

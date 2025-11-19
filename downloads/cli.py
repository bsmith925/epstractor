# downloads/cli.py

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.download import Downloader
from rich.console import Console
from rich.logging import RichHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download datasets defined in YAML configs."
    )
    parser.add_argument(
        "source",
        nargs="?",
        help="Source name (config file name without .yaml). Omit to use --all.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all configs in the config directory.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=Path(__file__).parent / "configs",
        help="Directory containing {source}.yaml configs (default: downloads/configs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "datasets",
        help="Root directory for downloaded datasets (default: downloads/datasets)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files instead of skipping them.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    parser.add_argument(
        "--service-account-file",
        type=Path,
        help=(
            "Path to Google service account JSON file. "
            "Alternative to GOOGLE_SERVICE_ACCOUNT_FILE env var. "
            "Recommended for automated access. Requires sharing folder with service account email. "
            "See: https://console.cloud.google.com/iam-admin/serviceaccounts"
        ),
    )
    parser.add_argument(
        "--credentials-file",
        type=Path,
        default=Path("credentials.json"),
        help=(
            "Path to OAuth2 credentials JSON file (default: credentials.json). "
            "Alternative to GOOGLE_CREDENTIALS_FILE env var. "
            "Works for public folders. Download from Google Cloud Console. "
            "See: https://console.cloud.google.com/apis/credentials"
        ),
    )
    parser.add_argument(
        "--api-key",
        help=(
            "Google API key (not recommended, may be blocked). "
            "Alternative to GOOGLE_API_KEY env var or .env file. "
            "The Drive API download flow does not support API keys - use OAuth2 credentials instead."
        ),
    )
    parser.add_argument(
        "--max-drive-workers",
        type=int,
        default=4,
        help="Maximum concurrent Google Drive downloads (default: 4).",
    )
    parser.add_argument(
        "--max-http-workers",
        type=int,
        default=8,
        help="Maximum concurrent HTTP downloads (default: 8).",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Enumerate Google Drive files and write manifest without downloading.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Skip downloads and verify local files against manifests.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip checksum/size verification after downloads (faster but unsafe).",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show progress bars (use --no-progress to disable).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.progress:
        console = Console()
        logging.basicConfig(
            level=getattr(logging, args.log_level),
            format="%(message)s",
            handlers=[
                RichHandler(
                    console=console,
                    rich_tracebacks=True,
                    show_time=True,
                    show_level=True,
                    show_path=False,
                    markup=True,
                )
            ],
        )
    else:
        console = None
        logging.basicConfig(
            level=getattr(logging, args.log_level),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )

    # Determine which configs to process
    if args.all:
        config_files = sorted(args.config_dir.glob("*.yaml"))
        if not config_files:
            logging.error("No YAML config files found in %s", args.config_dir)
            sys.exit(1)
        config_paths = config_files
    elif args.source:
        config_path = args.config_dir / f"{args.source}.yaml"
        if not config_path.exists():
            logging.error("Config file not found: %s", config_path)
            sys.exit(1)
        config_paths = [config_path]
    else:
        logging.error(
            "Either specify a source name or use --all to download all configs."
        )
        sys.exit(1)

    # Process each config
    failed = []
    for config_path in config_paths:
        source_name = config_path.stem
        logging.info("Processing config: %s", source_name)

        try:
            downloader = Downloader(
                config_path=config_path,
                base_output_dir=args.output_dir,
                api_key=args.api_key,
                service_account_file=args.service_account_file,
                credentials_file=args.credentials_file,
                max_http_workers=args.max_http_workers,
                max_drive_workers=args.max_drive_workers,
                manifest_only=args.manifest_only,
                verify_only=args.verify_only,
                verify_downloads=not args.skip_verify,
                use_progress=args.progress,
                console=console,
            )
            downloader.download_all(overwrite=args.overwrite)
        except Exception as e:
            logging.error("Failed to process config %s: %s", source_name, e)
            failed.append(source_name)
            continue

    if failed:
        logging.error(
            "Failed to download %d config(s): %s", len(failed), ", ".join(failed)
        )
        sys.exit(1)


if __name__ == "__main__":
    main()

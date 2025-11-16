# huggingface/upload_dataset.py

"""
Upload parquet files to HuggingFace Hub as a dataset with named subsets.

This script loads parquet files from subdirectories and pushes them to the Hub
as separate dataset splits/subsets that users can access independently.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from huggingface_hub import HfApi, login
from rich.console import Console
from rich.logging import RichHandler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload parquet datasets to HuggingFace Hub with named subsets"
    )
    parser.add_argument(
        "--repo-id",
        default="bsmith925/epstractor",
        help="HuggingFace repo ID (default: bsmith925/epstractor)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("huggingface/datasets/epstractor/data"),
        help="Directory containing subdirectories with parquet files (default: huggingface/datasets/epstractor/data)",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN"),
        help="HuggingFace token (optional if already logged in with `huggingface-cli login`)",
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Target branch (default: main)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be uploaded without uploading",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    return parser.parse_args()


def discover_subsets(data_dir: Path) -> dict[str, str]:
    """
    Discover parquet file subsets in data_dir.
    
    Returns a dict mapping subset names to glob patterns.
    """
    subsets = {}
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Look for subdirectories containing parquet files
    for subdir in sorted(data_dir.iterdir()):
        if subdir.is_dir():
            parquet_files = list(subdir.glob("*.parquet"))
            if parquet_files:
                # Use glob pattern relative to data_dir
                pattern = f"{subdir.name}/*.parquet"
                subsets[subdir.name] = str(data_dir / pattern)
                logging.info(
                    f"Found subset '{subdir.name}' with {len(parquet_files)} parquet file(s)"
                )
    
    return subsets


def main() -> None:
    args = parse_args()
    
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
    
    logger = logging.getLogger(__name__)
    
    # Get token for authentication
    token = args.token
    if not token:
        # Try to read from cached token file
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            token = token_path.read_text().strip()
            logger.info("Using cached HuggingFace credentials")
        else:
            if not args.dry_run:
                logger.error(
                    "No HuggingFace token found. Please either:\n"
                    "  1. Set HF_TOKEN environment variable, or\n"
                    "  2. Run: uv run python -c 'from huggingface_hub import login; login()'"
                )
                return
    else:
        logger.info("Using provided token")
    
    # Discover subsets
    logger.info(f"Scanning {args.data_dir} for parquet subsets...")
    try:
        data_files = discover_subsets(args.data_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    if not data_files:
        logger.warning(f"No parquet subsets found in {args.data_dir}")
        return
    
    # Count files for each subset
    subset_file_counts = {}
    for subset_name in data_files.keys():
        subset_dir = args.data_dir / subset_name
        parquet_files = list(subset_dir.glob("*.parquet"))
        subset_file_counts[subset_name] = len(parquet_files)
    
    # Display what will be uploaded
    console.print("\n[bold cyan]Dataset Upload Configuration[/bold cyan]")
    console.print(f"[bold]Repository:[/bold] {args.repo_id}")
    console.print(f"[bold]Branch:[/bold] {args.branch}")
    console.print(f"[bold]Private:[/bold] {args.private}")
    console.print(f"\n[bold]Subsets to upload:[/bold]")
    for subset_name, count in subset_file_counts.items():
        console.print(f"  • {subset_name}: {count} parquet file(s)")
    
    if args.dry_run:
        console.print("\n[yellow]Dry run mode - no upload will be performed[/yellow]")
        return
    
    # Upload parquet files directly to Hub (no caching/loading)
    logger.info(f"\nUploading parquet files directly to {args.repo_id}...")
    logger.info("This avoids disk space issues by uploading files as-is")
    
    try:
        # Initialize API with token
        api = HfApi(token=token)
        
        # Create the repo if it doesn't exist
        logger.info(f"Creating/checking repository {args.repo_id}...")
        api.create_repo(
            repo_id=args.repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,
        )
        logger.info(f"✓ Repository {args.repo_id} is ready")
        
        # Upload each subset's parquet files
        for subset_name in data_files.keys():
            subset_dir = args.data_dir / subset_name
            parquet_files = sorted(subset_dir.glob("*.parquet"))
            
            logger.info(f"\nUploading {len(parquet_files)} file(s) for subset '{subset_name}'...")
            
            # Upload all parquet files for this subset to data/{subset_name}/
            api.upload_folder(
                folder_path=str(subset_dir),
                repo_id=args.repo_id,
                repo_type="dataset",
                path_in_repo=f"data/{subset_name}",
                revision=args.branch,
                allow_patterns="*.parquet",
                commit_message=f"Upload {subset_name} parquet files",
            )
            
            logger.info(f"✓ Uploaded {subset_name}")
        
        # Also check if there's a README.md to upload
        readme_path = args.data_dir.parent / "README.md"
        if readme_path.exists():
            logger.info("\nUploading README.md...")
            api.upload_file(
                path_or_fileobj=str(readme_path),
                path_in_repo="README.md",
                repo_id=args.repo_id,
                repo_type="dataset",
                revision=args.branch,
                commit_message="Upload dataset card",
            )
            logger.info("✓ Uploaded README.md")
        
        console.print(f"\n[bold green]✓ Upload complete![/bold green]")
        console.print(f"\nYour dataset is now available at:")
        console.print(f"  https://huggingface.co/datasets/{args.repo_id}")
        console.print(f"\nLoad it with:")
        console.print(f'  [cyan]from datasets import load_dataset[/cyan]')
        console.print(f'  [cyan]ds = load_dataset("{args.repo_id}")[/cyan]')
        console.print(f"\nAccess individual subsets:")
        for subset_name in data_files.keys():
            console.print(f'  [cyan]ds["{subset_name}"][/cyan]')
        
    except Exception as e:
        logger.error(f"Failed to upload to hub: {e}", exc_info=True)
        return


if __name__ == "__main__":
    main()


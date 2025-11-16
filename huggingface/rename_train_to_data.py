# huggingface/rename_to_no_split.py

"""
Rename parquet files from train-*.parquet to {source_name}-*.parquet pattern.

This is a one-time migration script to remove the split concept entirely.
For document archives, each source/subset is a config without splits,
making the API cleaner: load_dataset(repo, name=source) returns Dataset directly.
"""

import argparse
import logging
from pathlib import Path
from huggingface_hub import HfApi
from rich.console import Console
from rich.logging import RichHandler


def rename_local_files(data_dir: Path, dry_run: bool = False) -> dict[str, list[tuple[Path, Path]]]:
    """
    Rename local parquet files from train-*.parquet to {source_name}-*.parquet.
    
    This removes the split concept entirely - each source becomes a config without splits.
    
    Returns dict mapping subset names to list of (old_path, new_path) tuples.
    """
    console = Console()
    renamed = {}
    
    if not data_dir.exists():
        console.print(f"[red]Error: Data directory not found: {data_dir}[/red]")
        return renamed
    
    # Process each subset directory
    for subset_dir in sorted(data_dir.iterdir()):
        if not subset_dir.is_dir():
            continue
        
        # Find train-*.parquet files
        train_files = list(subset_dir.glob("train-*.parquet"))
        if not train_files:
            continue
        
        subset_name = subset_dir.name
        renamed[subset_name] = []
        
        console.print(f"\n[cyan]{subset_name}:[/cyan] {len(train_files)} file(s)")
        
        for old_path in sorted(train_files):
            # Replace "train-" prefix with source name
            new_name = old_path.name.replace("train-", f"{subset_name}-", 1)
            new_path = old_path.parent / new_name
            
            renamed[subset_name].append((old_path, new_path))
            
            if dry_run:
                console.print(f"  [dim]Would rename:[/dim] {old_path.name} → {new_name}")
            else:
                old_path.rename(new_path)
                console.print(f"  [green]✓[/green] {old_path.name} → {new_name}")
    
    return renamed


def update_hub(
    repo_id: str,
    data_dir: Path,
    token: str | None,
    branch: str = "main",
    dry_run: bool = False,
) -> None:
    """
    Update HuggingFace Hub:
    1. Delete old train-*.parquet files
    2. Upload new data-*.parquet files (Git LFS will dedupe content)
    """
    console = Console()
    logger = logging.getLogger(__name__)
    
    if dry_run:
        console.print("\n[yellow]Dry run - no Hub changes will be made[/yellow]")
        return
    
    console.print(f"\n[bold]Updating HuggingFace Hub: {repo_id}[/bold]")
    
    api = HfApi(token=token)
    
    # Get list of files in repo
    logger.info("Fetching file list from Hub...")
    try:
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", revision=branch)
        train_files = [f for f in files if "/train-" in f and f.endswith(".parquet")]
        
        console.print(f"Found {len(train_files)} train-*.parquet file(s) on Hub")
    except Exception as e:
        logger.error(f"Failed to list repo files: {e}")
        return
    
    if not train_files:
        console.print("[yellow]No train-*.parquet files found on Hub[/yellow]")
        return
    
    # Step 1: Delete old train-*.parquet files from Hub
    console.print("\n[cyan]Step 1: Deleting old train-*.parquet files from Hub...[/cyan]")
    for file_path in train_files:
        try:
            api.delete_file(
                path_in_repo=file_path,
                repo_id=repo_id,
                repo_type="dataset",
                revision=branch,
                commit_message=f"Remove old file: {file_path}",
            )
            console.print(f"  [red]✗[/red] Deleted: {file_path}")
        except Exception as e:
            logger.error(f"Failed to delete {file_path}: {e}")
    
    # Step 2: Upload new {source}-*.parquet files
    console.print("\n[cyan]Step 2: Uploading renamed parquet files...[/cyan]")
    console.print("[dim]Git LFS will dedupe identical content - only filenames change[/dim]")
    
    for subset_dir in sorted(data_dir.iterdir()):
        if not subset_dir.is_dir():
            continue
        
        subset_name = subset_dir.name
        source_files = list(subset_dir.glob(f"{subset_name}-*.parquet"))
        if not source_files:
            continue
        
        logger.info(f"Uploading {len(source_files)} file(s) for {subset_name}...")
        
        try:
            api.upload_folder(
                folder_path=str(subset_dir),
                repo_id=repo_id,
                repo_type="dataset",
                path_in_repo=f"data/{subset_name}",
                revision=branch,
                allow_patterns=f"{subset_name}-*.parquet",
                commit_message=f"Upload renamed parquet files for {subset_name} (no split)",
            )
            console.print(f"  [green]✓[/green] Uploaded {subset_name} ({len(source_files)} files)")
        except Exception as e:
            logger.error(f"Failed to upload {subset_name}: {e}")
    
    console.print("\n[bold green]✓ Hub update complete![/bold green]")
    console.print("\nGit LFS should have deduped the content - check repo size to confirm.")


def main():
    parser = argparse.ArgumentParser(
        description="Rename parquet files from train-*.parquet to {source_name}-*.parquet (removes splits)"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("huggingface/datasets/epstractor/data"),
        help="Local data directory (default: huggingface/datasets/epstractor/data)",
    )
    parser.add_argument(
        "--repo-id",
        default="bsmith925/epstractor",
        help="HuggingFace repo ID (default: bsmith925/epstractor)",
    )
    parser.add_argument(
        "--token",
        help="HuggingFace token (or set HF_TOKEN env var)",
    )
    parser.add_argument(
        "--branch",
        default="main",
        help="Branch to update (default: main)",
    )
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Only rename local files, don't update Hub",
    )
    parser.add_argument(
        "--hub-only",
        action="store_true",
        help="Only update Hub (assumes local files already renamed)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    
    args = parser.parse_args()
    
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
    
    console.print("\n[bold]Parquet File Rename: train → source_name (remove splits)[/bold]\n")
    
    if args.dry_run:
        console.print("[yellow]DRY RUN MODE - No changes will be made[/yellow]\n")
    
    # Step 1: Rename local files (unless --hub-only)
    if not args.hub_only:
        console.print("[bold cyan]Renaming local files...[/bold cyan]")
        renamed = rename_local_files(args.data_dir, dry_run=args.dry_run)
        
        if renamed:
            total = sum(len(files) for files in renamed.values())
            console.print(f"\n[green]✓ Renamed {total} file(s) across {len(renamed)} subset(s)[/green]")
        else:
            console.print("\n[yellow]No train-*.parquet files found to rename[/yellow]")
            return
    
    # Step 2: Update Hub (unless --local-only or --dry-run)
    if not args.local_only and not args.dry_run:
        import os
        token = args.token or os.environ.get("HF_TOKEN")
        
        if not token:
            console.print(
                "\n[yellow]No HuggingFace token provided.[/yellow]\n"
                "To update the Hub, either:\n"
                "  1. Set HF_TOKEN environment variable, or\n"
                "  2. Pass --token argument\n\n"
                "Or use --local-only to skip Hub update."
            )
            return
        
        update_hub(
            repo_id=args.repo_id,
            data_dir=args.data_dir,
            token=token,
            branch=args.branch,
            dry_run=args.dry_run,
        )
    elif args.local_only:
        console.print("\n[dim]Skipping Hub update (--local-only)[/dim]")
    
    console.print("\n[bold]Done![/bold]\n")


if __name__ == "__main__":
    main()


# huggingface/filter_parquet.py

"""
Remove specific rows from parquet files based on path patterns.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow.parquet as pq
from rich.console import Console
from rich.progress import track


def filter_parquet_files(
    data_dir: Path,
    exclude_patterns: list[str],
    dry_run: bool = False,
    fast_mode: bool = True,
) -> None:
    """
    Remove rows from parquet files where the 'path' column matches any exclude pattern.
    
    Args:
        data_dir: Directory containing subdirectories with parquet files
        exclude_patterns: List of patterns to match in the 'path' column (e.g., 'dedupe_report.json')
        dry_run: If True, show what would be removed without modifying files
        fast_mode: If True, only check first and last parquet in each subset (much faster)
    """
    console = Console()
    
    # Find all parquet files
    all_parquet_files = list(data_dir.rglob("*.parquet"))
    
    if not all_parquet_files:
        console.print(f"[yellow]No parquet files found in {data_dir}[/yellow]")
        return
    
    # Fast mode: only check first/last file in each subset directory
    if fast_mode:
        subsets = {}
        for f in all_parquet_files:
            subset_dir = f.parent
            if subset_dir not in subsets:
                subsets[subset_dir] = []
            subsets[subset_dir].append(f)
        
        # For each subset, take first and last file (sorted by name)
        parquet_files = []
        for subset_dir, files in subsets.items():
            sorted_files = sorted(files)
            if len(sorted_files) == 1:
                parquet_files.extend(sorted_files)
            else:
                parquet_files.extend([sorted_files[0], sorted_files[-1]])
        
        console.print(f"\n[bold cyan]Filtering Parquet Files (Fast Mode)[/bold cyan]")
        console.print(f"[dim]Checking first/last files in each subset only[/dim]")
    else:
        parquet_files = all_parquet_files
        console.print(f"\n[bold cyan]Filtering Parquet Files[/bold cyan]")
    
    console.print(f"Directory: [dim]{data_dir}[/dim]")
    console.print(f"Exclude patterns: [yellow]{', '.join(exclude_patterns)}[/yellow]")
    console.print(f"Checking {len(parquet_files)} of {len(all_parquet_files)} parquet file(s)\n")
    
    if dry_run:
        console.print("[yellow]DRY RUN - No files will be modified[/yellow]\n")
    
    total_removed = 0
    files_modified = 0
    
    for parquet_file in track(parquet_files, description="Processing files"):
        try:
            # Read the parquet file
            table = pq.read_table(parquet_file)
            original_rows = len(table)
            
            # Get the path column
            if 'path' not in table.column_names:
                console.print(f"[yellow]Skipping {parquet_file.name}: no 'path' column[/yellow]")
                continue
            
            paths = table['path'].to_pylist()
            
            # Find rows to keep (inverse of exclude patterns)
            keep_mask = [
                not any(pattern in path for pattern in exclude_patterns)
                for path in paths
            ]
            
            rows_to_remove = original_rows - sum(keep_mask)
            
            if rows_to_remove > 0:
                files_modified += 1
                total_removed += rows_to_remove
                
                console.print(
                    f"[cyan]{parquet_file.relative_to(data_dir)}[/cyan]: "
                    f"removing {rows_to_remove} row(s) "
                    f"({original_rows} → {sum(keep_mask)})"
                )
                
                if not dry_run:
                    # Filter the table
                    filtered_table = table.filter(keep_mask)
                    
                    # Write back to the same file
                    pq.write_table(
                        filtered_table,
                        parquet_file,
                        compression='snappy',
                    )
        
        except Exception as e:
            console.print(f"[red]Error processing {parquet_file.name}: {e}[/red]")
    
    console.print(f"\n[bold]Summary[/bold]")
    console.print(f"Files modified: [yellow]{files_modified}[/yellow]")
    console.print(f"Total rows removed: [yellow]{total_removed}[/yellow]")
    
    if dry_run:
        console.print(f"\n[yellow]This was a dry run. Run without --dry-run to apply changes.[/yellow]")
    else:
        console.print(f"\n[green]✓ Filtering complete![/green]")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove rows from parquet files based on path patterns"
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("huggingface/datasets/epstractor/data"),
        help="Directory containing parquet files (default: huggingface/datasets/epstractor/data)",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=["dedupe_report.json"],
        help="Patterns to exclude from path column (default: dedupe_report.json)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be removed without modifying files",
    )
    parser.add_argument(
        "--full-scan",
        action="store_true",
        help="Check all files instead of just first/last in each subset (slower)",
    )
    
    args = parser.parse_args()
    
    if not args.data_dir.exists():
        print(f"Error: Directory not found: {args.data_dir}")
        return 1
    
    filter_parquet_files(
        data_dir=args.data_dir,
        exclude_patterns=args.exclude,
        dry_run=args.dry_run,
        fast_mode=not args.full_scan,
    )
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())


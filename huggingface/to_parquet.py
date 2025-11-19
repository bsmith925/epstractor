# huggingface/to_parquet.py

"""Convert datasets to Parquet format using PyArrow."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from rich.console import Console
from rich.progress import track
from rich.table import Table

# Ensure output isn't buffered
sys.stdout.reconfigure(line_buffering=True)


def convert_to_parquet(
    source_dir: Path,
    output_path: Path,
    max_shard_bytes: int = 500 * 1024 * 1024,
    exclude_patterns: list[str] | None = None,
    large_files_dir: Path | None = None,
):
    """
    Convert directory of files to Parquet with size-based sharding.
    
    Features:
    - Collect all files
    - Build PyArrow batches
    - Write shards when size limit reached
    - Split files >2GB into chunks to work around PyArrow's 2GB limit
    
    Large files (>2GB) are automatically split into 1GB chunks and stored
    across multiple rows. Use chunk_index and total_chunks columns to
    reassemble them.
    """
    console = Console()
    exclude_patterns = exclude_patterns or ['.manifest.json', '__MACOSX', '.DS_Store']
    
    # PyArrow's limit for a single binary value
    MAX_FILE_SIZE = 2_000_000_000  # ~2GB (slightly under 2^31 bytes)
    CHUNK_SIZE = 1_000_000_000  # 1GB chunks for files >2GB
    
    console.print(f"\n[bold cyan]{source_dir.name}[/bold cyan]")
    console.print(f"  Source: [dim]{source_dir}[/dim]")
    console.print(f"  Output: [dim]{output_path.parent}[/dim]")
    console.print(f"  Shard size: [yellow]{max_shard_bytes / 1_048_576:.0f} MB[/yellow]")
    console.print(f"  Chunk size for large files: [yellow]{CHUNK_SIZE / 1_048_576:.0f} MB[/yellow]\n")
    
    # Collect files
    console.print("[cyan]Scanning files...[/cyan]")
    files = []
    for f in source_dir.rglob('*'):
        if f.is_file() and not any(p in str(f) for p in exclude_patterns):
            files.append(f)
    
    if not files:
        console.print("[yellow]No files found[/yellow]")
        return
    
    console.print(f"[green]Found {len(files):,} files[/green]\n")
    
    # Define schema
    schema = pa.schema([
        ('path', pa.string()),
        ('source', pa.string()),
        ('file_type', pa.string()),
        ('file_size', pa.int64()),
        ('extension', pa.string()),
        ('content', pa.binary()),
        ('content_available', pa.bool_()),  # True if full content is available
        ('chunk_index', pa.int32()),  # 0 for non-chunked files, 0..N for chunked
        ('total_chunks', pa.int32()),  # 1 for non-chunked files, N+1 for chunked
    ])
    
    # Process files and write shards
    console.print("[cyan]Converting to parquet...[/cyan]")
    
    shard_num = 0
    current_shard_data = []
    current_shard_size = 0
    total_bytes = 0
    chunked_files = 0
    
    for file_path in track(files, description="Processing"):
        try:
            # Check file size first
            size = file_path.stat().st_size
            total_bytes += size
            
            # Classify file type
            ext = file_path.suffix.lower()
            if ext in {'.jpg', '.jpeg', '.tif', '.tiff', '.png'}:
                file_type = "image"
            elif ext in {'.txt', '.md', '.json'}:
                file_type = "text"
            elif ext in {'.wav', '.mp3'}:
                file_type = "audio"
            elif ext in {'.mp4', '.mov'}:
                file_type = "video"
            elif ext in {'.pdf', '.xls', '.xlsx'}:
                file_type = "document"
            else:
                file_type = "other"
            
            # Handle large files (>2GB) - split into chunks
            if size > MAX_FILE_SIZE:
                chunked_files += 1
                with open(file_path, 'rb') as f:
                    chunk_index = 0
                    while True:
                        chunk = f.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        
                        # Calculate total chunks (ceiling division)
                        total_chunks = (size + CHUNK_SIZE - 1) // CHUNK_SIZE
                        
                        current_shard_data.append({
                            'path': str(file_path.relative_to(source_dir)),
                            'source': source_dir.name,
                            'file_type': file_type,
                            'file_size': size,
                            'extension': ext,
                            'content': chunk,
                            'content_available': True,
                            'chunk_index': chunk_index,
                            'total_chunks': total_chunks,
                        })
                        
                        current_shard_size += len(chunk)
                        chunk_index += 1
                        
                        # Write shard if size exceeded
                        if current_shard_size >= max_shard_bytes:
                            shard_num += 1
                            write_shard(output_path, shard_num, current_shard_data, schema)
                            current_shard_data = []
                            current_shard_size = 0
            else:
                # Normal file - read in one go
                content = file_path.read_bytes()
                
                current_shard_data.append({
                    'path': str(file_path.relative_to(source_dir)),
                    'source': source_dir.name,
                    'file_type': file_type,
                    'file_size': size,
                    'extension': ext,
                    'content': content,
                    'content_available': True,
                    'chunk_index': 0,
                    'total_chunks': 1,
                })
                
                current_shard_size += size
                
                # Write shard if size exceeded
                if current_shard_size >= max_shard_bytes:
                    shard_num += 1
                    write_shard(output_path, shard_num, current_shard_data, schema)
                    current_shard_data = []
                    current_shard_size = 0
                
        except Exception as e:
            console.print(f"[yellow]Skipped {file_path.name}: {e}[/yellow]")
    
    # Write final shard
    if current_shard_data:
        shard_num += 1
        write_shard(output_path, shard_num, current_shard_data, schema)
    
    # Calculate total shards for consistent naming
    if shard_num > 1:
        rename_shards(output_path, shard_num)
    
    console.print(f"\n[green]✓ Created {shard_num} shard(s)[/green]")
    console.print(f"  Total size: [yellow]{total_bytes / 1_073_741_824:.2f} GB[/yellow]")
    console.print(f"  Files: [yellow]{len(files):,}[/yellow]")
    
    if chunked_files > 0:
        console.print(f"  [cyan]Large files (>2GB, split into chunks): {chunked_files}[/cyan]")
    
    console.print()
    
    return {
        "name": source_dir.name,
        "total_files": len(files),
        "total_bytes": total_bytes,
        "shard_count": shard_num,
    }


def write_shard(output_path: Path, shard_num: int, data: list[dict], schema: pa.Schema):
    """Write a single parquet shard."""
    # Temporary filename (will rename later with correct total)
    if shard_num == 1:
        filename = output_path
    else:
        filename = output_path.with_name(f"{output_path.stem}-{shard_num:05d}.parquet")
    
    filename.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to PyArrow table
    table = pa.table({
        'path': [d['path'] for d in data],
        'source': [d['source'] for d in data],
        'file_type': [d['file_type'] for d in data],
        'file_size': [d['file_size'] for d in data],
        'extension': [d['extension'] for d in data],
        'content': [d['content'] for d in data],
        'content_available': [d['content_available'] for d in data],
        'chunk_index': [d['chunk_index'] for d in data],
        'total_chunks': [d['total_chunks'] for d in data],
    }, schema=schema)
    
    # Write parquet
    pq.write_table(table, filename, compression='snappy')


def rename_shards(output_path: Path, total_shards: int):
    """Rename shards with consistent naming: 00000-of-00003.parquet"""
    # If we only wrote one shard, rename it to 00000-of-00001
    if total_shards == 1:
        if output_path.exists():
            new_name = output_path.with_name(f"{output_path.stem}-00000-of-00001.parquet")
            output_path.rename(new_name)
        return
    
    # First handle the initial file (no number) -> 00000
    first_file = output_path
    if first_file.exists():
        new_name = output_path.with_name(f"{output_path.stem}-00000-of-{total_shards:05d}.parquet")
        first_file.rename(new_name)
    
    # Then rename numbered files (1-indexed temp names -> 0-indexed final names)
    for i in range(2, total_shards + 1):
        old_name = output_path.with_name(f"{output_path.stem}-{i:05d}.parquet")
        new_name = output_path.with_name(f"{output_path.stem}-{(i-1):05d}-of-{total_shards:05d}.parquet")
        if old_name.exists():
            old_name.rename(new_name)


def main():
    """CLI for parquet conversion."""
    parser = argparse.ArgumentParser(description="Convert datasets to Parquet")
    parser.add_argument("config", help="Config name from huggingface/datasets/configs/")
    parser.add_argument("--configs-dir", type=Path, default=Path("huggingface/datasets/configs"))
    parser.add_argument("--shard-size-mb", type=int, default=500, help="Shard size in MB")
    args = parser.parse_args()
    
    console = Console()
    
    # Load config
    config_path = args.configs_dir / f"{args.config}.yaml"
    if not config_path.exists():
        console.print(f"[red]Config not found:[/red] {config_path}")
        return 1
    
    with config_path.open() as f:
        config = yaml.safe_load(f)
    
    # Get settings
    parquet_config = config.get("parquet", {})
    output_dir = Path(parquet_config.get("output_dir", "epstein_all/data"))
    max_shard_bytes = args.shard_size_mb * 1024 * 1024
    exclude_patterns = parquet_config.get("exclude_patterns", [".manifest.json", "__MACOSX", ".DS_Store"])
    
    # Find datasets
    roots = config.get("roots", ["downloads/datasets"])
    include = config.get("include", [])
    
    datasets = []
    for name in include:
        for root in roots:
            path = Path(root) / name
            if path.exists() and path.is_dir():
                datasets.append(path)
                break
    
    if not datasets:
        console.print("[yellow]No datasets found[/yellow]")
        return 0
    
    # Show summary
    console.print(f"\n[bold]Parquet Conversion: {args.config}[/bold]\n")
    
    table = Table()
    table.add_column("Dataset", style="cyan")
    table.add_column("Path", style="dim")
    for ds in datasets:
        table.add_row(ds.name, str(ds))
    console.print(table)
    
    # Convert each
    all_stats = []
    for ds_path in datasets:
        try:
            stats = convert_to_parquet(
                source_dir=ds_path,
                output_path=output_dir / ds_path.name / f"{ds_path.name}.parquet",
                max_shard_bytes=max_shard_bytes,
                exclude_patterns=exclude_patterns,
            )
            all_stats.append(stats)
        except Exception as e:
            console.print(f"[red]Failed {ds_path.name}:[/red] {e}")
    
    # Final summary
    if all_stats:
        console.print("[bold]Summary[/bold]\n")
        summary = Table()
        summary.add_column("Dataset", style="cyan")
        summary.add_column("Files", justify="right")
        summary.add_column("Size", justify="right")
        summary.add_column("Shards", justify="right")
        
        for s in all_stats:
            summary.add_row(
                s["name"],
                f"{s['total_files']:,}",
                f"{s['total_bytes'] / 1_073_741_824:.2f} GB",
                str(s["shard_count"]),
            )
        
        console.print(summary)
        console.print(f"\n[green]✓ All done![/green]\n")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

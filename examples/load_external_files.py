# examples/load_external_files.py

"""
Example: Working with external files (large files not embedded in parquet).

For files >2GB, the content is stored separately in the repo's large_files/ directory
and referenced via the external_file column.
"""

from datasets import load_dataset
from huggingface_hub import hf_hub_download
from pathlib import Path


def load_file_content(row: dict, repo_id: str = "public-records-research/epstractor-raw") -> bytes:
    """
    Load file content, handling both embedded and external files.
    
    Args:
        row: Dataset row
        repo_id: HuggingFace repository ID
        
    Returns:
        File content as bytes
    """
    if row['content_available']:
        # Content is embedded in parquet
        return row['content']
    elif row['external_file']:
        # Download external file
        file_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=row['external_file']
        )
        with open(file_path, 'rb') as f:
            return f.read()
    else:
        raise ValueError(f"Content not available for {row['path']}")


def example_1_list_external_files():
    """List all files that are stored externally."""
    print("=" * 70)
    print("Example 1: List External Files")
    print("=" * 70)
    
    # Load the dataset
    ds = load_dataset(
        "public-records-research/epstractor-raw",
        name="house_doj_2025_09",
        split="train"
    )
    
    # Filter for external files
    external = ds.filter(lambda x: x['external_file'] is not None)
    
    print(f"\nFound {len(external)} external file(s):\n")
    
    for row in external:
        size_gb = row['file_size'] / 1_073_741_824
        print(f"{row['path']}")
        print(f"  Size: {size_gb:.2f} GB")
        print(f"  Type: {row['file_type']} ({row['extension']})")
        print(f"  External path: {row['external_file']}")
        print()


def example_2_load_external_file():
    """Load an external file's content."""
    print("\n" + "=" * 70)
    print("Example 2: Load External File Content")
    print("=" * 70)
    
    # Load the dataset
    ds = load_dataset(
        "public-records-research/epstractor-raw",
        name="house_doj_2025_09",
        split="train"
    )
    
    # Find an external file
    external = ds.filter(lambda x: x['external_file'] is not None)
    
    if len(external) == 0:
        print("\nNo external files in this subset.")
        return
    
    # Load the first external file
    row = external[0]
    print(f"\nLoading: {row['path']}")
    print(f"Size: {row['file_size'] / 1_073_741_824:.2f} GB\n")
    
    # Download and get local path
    file_path = hf_hub_download(
        repo_id="public-records-research/epstractor-raw",
        repo_type="dataset",
        filename=row['external_file']
    )
    
    print(f"Downloaded to: {file_path}")
    print(f"Actual size: {Path(file_path).stat().st_size / 1_073_741_824:.2f} GB")
    print("\nYou can now use this file with your preferred video player or library.")


def example_3_selective_download():
    """Show how to download only the files you need."""
    print("\n" + "=" * 70)
    print("Example 3: Selective Download Pattern")
    print("=" * 70)
    
    # Load the dataset
    ds = load_dataset(
        "public-records-research/epstractor-raw",
        name="house_doj_2025_09",
        split="train"
    )
    
    print("\nDataset statistics:")
    print(f"  Total files: {len(ds):,}")
    
    # Count files by content availability
    embedded_count = len(ds.filter(lambda x: x['content_available']))
    external_count = len(ds.filter(lambda x: x['external_file'] is not None))
    
    print(f"  Embedded files: {embedded_count:,}")
    print(f"  External files: {external_count:,}")
    
    # Show size breakdown
    total_embedded = sum(r['file_size'] for r in ds if r['content_available'])
    total_external = sum(r['file_size'] for r in ds if r['external_file'])
    
    print(f"\nSize breakdown:")
    print(f"  Embedded: {total_embedded / 1_073_741_824:.2f} GB (downloaded automatically)")
    print(f"  External: {total_external / 1_073_741_824:.2f} GB (downloaded on-demand)")
    
    print("\nBenefit: You only download the large files if you actually need them!")


def example_4_unified_loader():
    """Unified approach that handles both embedded and external files."""
    print("\n" + "=" * 70)
    print("Example 4: Unified Loader (works for any file)")
    print("=" * 70)
    
    ds = load_dataset(
        "public-records-research/epstractor-raw",
        name="house_doj_2025_09",
        split="train"
    )
    
    # Take a mix of files
    print("\nLoading various files...\n")
    
    for i, row in enumerate(ds.select([0, 100, 200])):  # Sample a few files
        print(f"{i+1}. {row['path']}")
        print(f"   Size: {row['file_size'] / 1_048_576:.2f} MB")
        
        try:
            content = load_file_content(row)
            print(f"   ✓ Loaded {len(content):,} bytes")
        except Exception as e:
            print(f"   ✗ Error: {e}")
        print()


if __name__ == "__main__":
    # Run examples
    example_1_list_external_files()
    # example_2_load_external_file()  # Uncomment to test downloading
    # example_3_selective_download()
    # example_4_unified_loader()


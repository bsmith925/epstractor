# examples/load_subsets.py

"""
Example: Loading and working with epstractor dataset subsets.

This demonstrates how users can load and work with the different
subsets of the epstractor dataset.
"""

from datasets import load_dataset, DatasetDict, concatenate_datasets


def example_1_load_all():
    """Load all subsets as a DatasetDict."""
    print("=" * 70)
    print("Example 1: Load All Subsets")
    print("=" * 70)
    
    # Load all subsets (returns {subset_name: Dataset})
    ds = load_dataset("bsmith925/epstractor")
    print(ds)
    print(f"\nAvailable subsets: {list(ds.keys())}")
    
    for name, dataset in ds.items():
        # Each subset is a Dataset directly (no splits)
        print(f"\n{name}:")
        print(f"  Rows: {len(dataset):,}")
        print(f"  Columns: {dataset.column_names}")


def example_2_load_single():
    """Load just one subset."""
    print("\n" + "=" * 70)
    print("Example 2: Load Single Subset")
    print("=" * 70)
    
    # Load just the House DOJ data (returns Dataset directly, no split needed)
    house = load_dataset("bsmith925/epstractor", name="house_doj_2025_09")
    print(f"House DOJ dataset: {len(house):,} rows")
    
    # Look at first file
    first = house[0]
    print(f"\nFirst file:")
    print(f"  Path: {first['path']}")
    print(f"  Type: {first['file_type']}")
    print(f"  Size: {first['file_size']:,} bytes")
    print(f"  Extension: {first['extension']}")


def example_3_streaming():
    """Use streaming to avoid loading all data into memory."""
    print("\n" + "=" * 70)
    print("Example 3: Streaming for Large Subsets")
    print("=" * 70)
    
    # Stream the large Epstein estate dataset
    epstein_11 = load_dataset(
        "bsmith925/epstractor", 
        name="epstein_estate_2025_11",
        streaming=True
    )
    
    print("Streaming mode: Data is not loaded into memory")
    print("\nFirst 5 files:")
    
    # Process in batches without loading everything
    for i, row in enumerate(epstein_11):
        if i >= 5:
            break
        print(f"  {i+1}. {row['path']} ({row['file_type']})")


def example_4_filter():
    """Filter data from specific subsets."""
    print("\n" + "=" * 70)
    print("Example 4: Filtering by File Type")
    print("=" * 70)
    
    # Load one subset
    ds = load_dataset("bsmith925/epstractor", name="epstein_estate_2025_09")
    
    # Filter for PDF documents
    pdfs = ds.filter(lambda x: x['extension'] == '.pdf')
    print(f"Total files: {len(ds)}")
    print(f"PDF files: {len(pdfs)}")
    
    # Filter for large files (>10MB)
    large_files = ds.filter(lambda x: x['file_size'] > 10_000_000)
    print(f"Files >10MB: {len(large_files)}")


def example_5_train_eval_split():
    """Create train/eval splits from different subsets."""
    print("\n" + "=" * 70)
    print("Example 5: Train/Eval Split from Subsets")
    print("=" * 70)
    
    # Load specific subsets (returns Dataset directly)
    epstein_11 = load_dataset("bsmith925/epstractor", name="epstein_estate_2025_11")
    house = load_dataset("bsmith925/epstractor", name="house_doj_2025_09")
    
    # Create a train/eval split
    train_eval = DatasetDict({
        "train": epstein_11,
        "eval": house,
    })
    
    print(train_eval)
    print(f"\nTrain: {len(train_eval['train']):,} files")
    print(f"Eval: {len(train_eval['eval']):,} files")


def example_6_combine():
    """Combine multiple subsets."""
    print("\n" + "=" * 70)
    print("Example 6: Combining Subsets")
    print("=" * 70)
    
    # Load all subsets
    ds = load_dataset("bsmith925/epstractor")
    
    # Combine all into one dataset (direct access, no splits!)
    combined = concatenate_datasets([
        ds["epstein_estate_2025_09"],
        ds["epstein_estate_2025_11"],
        ds["house_doj_2025_09"],
    ])
    
    print(f"Combined dataset: {len(combined):,} total files")
    
    # Check distribution by source
    sources = {}
    for row in combined:
        source = row['source']
        sources[source] = sources.get(source, 0) + 1
    
    print("\nFiles by source:")
    for source, count in sorted(sources.items()):
        print(f"  {source}: {count:,}")


def example_7_batch_processing():
    """Process data in batches with streaming."""
    print("\n" + "=" * 70)
    print("Example 7: Batch Processing with Streaming")
    print("=" * 70)
    
    # Stream to avoid loading all data
    house = load_dataset(
        "bsmith925/epstractor", 
        name="house_doj_2025_09",
        streaming=True
    )
    
    # Process in batches of 100
    batch_size = 100
    processed = 0
    
    print(f"Processing in batches of {batch_size}...")
    
    # Use iter() to get batches
    batch = []
    for i, row in enumerate(house):
        batch.append(row)
        
        if len(batch) >= batch_size:
            # Process this batch
            print(f"  Batch {(i // batch_size) + 1}: {len(batch)} files")
            
            # Your processing here...
            # process_batch(batch)
            
            batch = []
            processed += batch_size
            
            # Just demo first few batches
            if processed >= 300:
                print(f"  ... (continuing for all files)")
                break


if __name__ == "__main__":
    print("\nEpstractor Dataset - Subset Loading Examples")
    print("=" * 70)
    print()
    print("NOTE: These examples assume the dataset is uploaded to:")
    print("      bsmith925/epstractor")
    print()
    print("Dataset structure:")
    print("  - Each subset (epstein_estate_2025_09, etc.) is a config")
    print("  - Each config is a complete dataset (no splits)")
    print("  - Load all: load_dataset('repo') → {config: Dataset}")
    print("  - Load one: load_dataset('repo', name='config')")
    print()
    
    # Run examples
    # Uncomment the ones you want to try:
    
    # example_1_load_all()
    # example_2_load_single()
    # example_3_streaming()
    # example_4_filter()
    # example_5_train_eval_split()
    # example_6_combine()
    # example_7_batch_processing()
    
    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)


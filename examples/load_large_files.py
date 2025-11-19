# examples/load_large_files.py

"""
Example: Working with large videos via the separate large_files config.

Large videos (>2GB) are stored as a separate config that can be joined
with the main datasets.
"""

from datasets import load_dataset
from huggingface_hub import hf_hub_download


def example_1_list_large_files():
    """List all large videos in the dataset."""
    print("=" * 70)
    print("Example 1: List Large Videos")
    print("=" * 70)
    
    # Load the large_files config
    videos = load_dataset(
        "public-records-research/epstractor-raw",
        name="large_files",
        split="train"
    )
    
    print(f"\nFound {len(videos)} large video(s):\n")
    
    for video in videos:
        size_gb = video['file_size'] / 1_073_741_824
        print(f"{video['file_id']}")
        print(f"  Source: {video['source']}")
        print(f"  Path: {video['path']}")
        print(f"  Size: {size_gb:.2f} GB")
        print(f"  Download from: {video['repo_path']}")
        print()


def example_2_join_with_main_dataset():
    """Show how videos relate to the main dataset."""
    print("\n" + "=" * 70)
    print("Example 2: Relating Videos to Main Dataset")
    print("=" * 70)
    
    # Load main dataset
    house_doj = load_dataset(
        "public-records-research/epstractor-raw",
        name="house_doj_2025_09",
        split="train"
    )
    
    # Load videos
    videos = load_dataset(
        "public-records-research/epstractor-raw",
        name="large_files",
        split="train"
    )
    
    print(f"\nMain dataset: {len(house_doj):,} files")
    print(f"Large videos: {len(videos)} files")
    
    # Show which videos came from house_doj
    house_videos = videos.filter(lambda x: x['source'] == 'house_doj_2025_09')
    
    print(f"\nVideos from house_doj_2025_09: {len(house_videos)}")
    for v in house_videos:
        print(f"  {v['file_id']} ({v['file_size'] / 1_073_741_824:.1f} GB)")


def example_3_download_video():
    """Download a large video file."""
    print("\n" + "=" * 70)
    print("Example 3: Download a Large Video")
    print("=" * 70)
    
    # Load videos config
    videos = load_dataset(
        "public-records-research/epstractor-raw",
        name="large_files",
        split="train"
    )
    
    # Get first video
    video = videos[0]
    
    print(f"\nDownloading: {video['file_id']}")
    print(f"Size: {video['file_size'] / 1_073_741_824:.2f} GB")
    print(f"This may take a while...\n")
    
    # Download via huggingface_hub
    local_path = hf_hub_download(
        repo_id="public-records-research/epstractor-raw",
        repo_type="dataset",
        filename=video['repo_path']
    )
    
    print(f"✓ Downloaded to: {local_path}")
    print(f"You can now open this file with any video player")


def example_4_selective_loading():
    """Only load metadata, download videos when needed."""
    print("\n" + "=" * 70)
    print("Example 4: Selective Loading Pattern")
    print("=" * 70)
    
    # Load all configs
    all_data = load_dataset("public-records-research/epstractor-raw")
    
    print("\nAvailable configs:")
    for config_name in all_data.keys():
        ds = all_data[config_name]
        if config_name == "large_files":
            total_size = sum(r['file_size'] for r in ds)
            print(f"  {config_name}: {len(ds)} videos ({total_size / 1_073_741_824:.1f} GB)")
        else:
            print(f"  {config_name}: {len(ds):,} files")
    
    print("\n✓ With this structure, you:")
    print("  - Download all metadata quickly (parquet files)")
    print("  - See what large videos exist")
    print("  - Only download videos you actually need")


def example_5_find_video_by_path():
    """Find the large video entry for a specific file."""
    print("\n" + "=" * 70)
    print("Example 5: Find Video by Original Path")
    print("=" * 70)
    
    videos = load_dataset(
        "public-records-research/epstractor-raw",
        name="large_files",
        split="train"
    )
    
    # Search by path substring
    search_term = "DOJ-OGR-00022169"
    
    matches = videos.filter(lambda x: search_term in x['path'])
    
    print(f"\nSearching for: '{search_term}'")
    print(f"Found {len(matches)} match(es):\n")
    
    for match in matches:
        print(f"  File ID: {match['file_id']}")
        print(f"  Full path: {match['path']}")
        print(f"  Size: {match['file_size'] / 1_073_741_824:.2f} GB")
        print(f"  Download: hf_hub_download(..., filename='{match['repo_path']}')")


if __name__ == "__main__":
    # Run examples
    example_1_list_large_files()
    example_2_join_with_main_dataset()
    # example_3_download_video()  # Uncomment to actually download (takes time!)
    example_4_selective_loading()
    example_5_find_video_by_path()


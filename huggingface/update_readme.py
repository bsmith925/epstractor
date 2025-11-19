# huggingface/update_readme.py

"""
Update README.md on HuggingFace Hub without re-uploading parquet files.
"""

from huggingface_hub import HfApi

api = HfApi()

api.upload_file(
    path_or_fileobj="huggingface/datasets/epstractor/README.md",
    path_in_repo="README.md",
    repo_id="public-records-research/epstractor-raw",
    repo_type="dataset",
    commit_message="docs: update README to reflect no-split structure"
)

print("✓ README.md uploaded successfully to public-records-research/epstractor-raw!")


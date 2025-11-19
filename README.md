# Epstractor

A tool for downloading, processing, and publishing large document archives to HuggingFace Hub.

## Overview

Epstractor downloads documents from Google Drive and other sources, converts them to Parquet format for efficient storage and access, and publishes them as HuggingFace datasets.

## Features

- **Download**: Fetch documents from Google Drive folders and HTTP sources
- **Deduplication**: Identify and handle duplicate files
- **Parquet Conversion**: Convert file collections to efficient Parquet format with sharding
- **HuggingFace Upload**: Publish datasets to HuggingFace Hub with proper metadata

## Quick Start

### Installation

This project uses `uv` for package management:

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync
```

### Configuration

1. **Google Drive Authentication** (for downloads):
   - Place your OAuth2 `credentials.json` in the project root
   - Or use a service account JSON file

2. **HuggingFace Token** (for uploads):
   ```bash
   export HF_TOKEN="hf_your_token_here"
   ```

### Common Tasks

#### Download Datasets

```bash
# Download a specific dataset
make download-epstein DATASET=epstein_estate_2025_11

# Or use the CLI directly
uv run download-datasets --source epstein_estate_2025_11
```

#### Deduplicate Files

```bash
make dedupe epstein_estate_2025_11
```

#### Convert to Parquet

```bash
make epstein_all parquet
```

#### Upload to HuggingFace

```bash
export HF_TOKEN="hf_your_token_here"
make hf-upload
```

## Project Structure

```
epstractor/
├── downloads/          # Download scripts and configs
│   ├── cli.py         # Download CLI entry point
│   ├── configs/       # Dataset download configurations (YAML)
│   └── dedupe.py      # Deduplication tool
├── huggingface/       # HuggingFace publishing tools
│   ├── datasets/
│   │   └── configs/   # Upload configurations (YAML)
│   ├── filter_parquet.py    # Filter parquet files
│   ├── to_parquet.py        # Parquet conversion
│   ├── upload_dataset.py    # HF upload script
│   └── validate_dataset.py  # Dataset validation
├── src/               # Core library code
│   └── download.py    # Download implementation
├── examples/          # Usage examples
│   └── load_subsets.py  # How to load datasets from HF Hub
├── Makefile           # Task automation
└── pyproject.toml     # Project dependencies
```

## Configuration Files

### Download Configs (`downloads/configs/*.yaml`)

Define sources to download:

```yaml
source: "epstein_estate_2025_11"
description: "Epstein Estate documents"
output_dir: "downloads/datasets"
items:
  - kind: gdrive_folder
    folder_id: "1ABC..."
    recursive: true
```

### Upload Configs (`huggingface/datasets/configs/*.yaml`)

Define datasets to upload:

```yaml
repo_id: "username/dataset-name"
roots:
  - "downloads/datasets"
include:
  - "epstein_estate_2025_09"
  - "epstein_estate_2025_11"
parquet:
  output_dir: "huggingface/datasets/epstractor/data"
  max_shard_size_mb: 500
```

## Documentation

- [UPLOAD_WORKFLOW.md](UPLOAD_WORKFLOW.md) - Detailed upload workflow and configuration

## Security

**Important**: Never commit credentials to git!

## License

See individual dataset licenses. This tool is provided as-is for working with publicly available government documents.


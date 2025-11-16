---
license: other
license_name: public-domain
license_link: https://www.usa.gov/government-works
task_categories:
  - other
language:
  - en
tags:
  - legal
  - documents
  - images
  - audio
  - video
  - archive
  - epstein
  - foia
size_categories:
  - 10K<n<100K
configs:
  - config_name: epstein_estate_2025_09
    data_files:
      - split: train
        path: data/epstein_estate_2025_09/*.parquet
  - config_name: epstein_estate_2025_11
    data_files:
      - split: train
        path: data/epstein_estate_2025_11/*.parquet
  - config_name: house_doj_2025_09
    data_files:
      - split: train
        path: data/house_doj_2025_09/*.parquet
---

# Epstractor: Epstein Archives Dataset

A comprehensive archive of documents, images, audio, and video files from multiple Epstein-related releases, including estate records and Department of Justice materials obtained through FOIA requests.

## Dataset Description

This dataset contains **59,420 files** totaling **115.23 GB** from three major document releases:

- **Epstein Estate 2025-09**: 5 files, 0.09 GB
- **Epstein Estate 2025-11**: 26,035 files, 36.56 GB  
- **House DOJ 2025-09**: 33,380 files, 78.58 GB

### File Types

- **Images**: ~56,418 files (JPG, TIF) - scanned documents and photographs
- **Text**: ~2,897 TXT files - OCR'd and extracted text
- **Audio**: 56 WAV files
- **Video**: 28 MP4/MOV files
- **Documents**: 14 PDF/XLS/XLSX files
- **Metadata**: JSON manifests

### Directory Structure

The dataset is organized by source for flexible loading:

```
data/
├── epstein_estate_2025_09/
│   └── train-00000-of-00001.parquet (1 shard, 0.09 GB)
├── epstein_estate_2025_11/
│   ├── train-00000-of-00075.parquet
│   ├── train-00001-of-00075.parquet
│   └── ... (75 shards, 36.56 GB)
└── house_doj_2025_09/
    ├── train-00000-of-00060.parquet
    ├── train-00001-of-00060.parquet
    └── ... (60 shards, 78.58 GB)
```

### Data Structure

The dataset is stored in Parquet format with the following schema:

| Column | Type | Description |
|--------|------|-------------|
| `path` | string | Relative file path within the source archive |
| `source` | string | Source dataset name (e.g., "epstein_estate_2025_11") |
| `file_type` | string | Classified type: image, text, audio, video, document, other |
| `file_size` | int64 | File size in bytes |
| `extension` | string | File extension (e.g., ".jpg", ".txt") |
| `content` | binary | Raw file bytes (null for files >2GB due to Arrow limits) |
| `content_available` | bool | Whether full content is available (false for files >2GB) |

## Source and Provenance

These materials were **officially released by the U.S. House Committee on Oversight and Accountability** via public Google Drive folders, making them freely accessible to the public without any access restrictions, paywalls, or distribution limitations.

**Official Release Channels:**
- House Oversight Committee public releases (September 2025) via Google Drive
- Epstein Estate public releases (September & November 2025) via Google Drive
- Department of Justice FOIA releases (2025)

All files were downloaded from publicly accessible sources with clear governmental intent to disseminate to the public. No proprietary or restricted materials are included.

**Chain of Custody:**
1. Congressional/Government agencies release materials publicly
2. Materials hosted on open Google Drive folders (no authentication required)
3. Downloaded and aggregated for research accessibility
4. Converted to Parquet format for efficient access
5. Published on HuggingFace for public research use

This is **not scraped, leaked, or proprietary content** - all materials were intentionally made public by official government sources.

## Usage

The dataset is organized by source in separate folders, allowing you to load all data or specific sources:

```python
from datasets import load_dataset

# Load ALL sources (136 shards, 59,420 files)
dataset = load_dataset("bsmith925/epstractor")

# Load ONLY a specific source using data_files pattern
estate_2025_11 = load_dataset("bsmith925/epstractor", 
                              data_files="data/epstein_estate_2025_11/*.parquet")

house_doj = load_dataset("bsmith925/epstractor",
                         data_files="data/house_doj_2025_09/*.parquet")

# Access a single file
row = dataset['train'][0]
print(f"Path: {row['path']}")
print(f"Source: {row['source']}")
print(f"Type: {row['file_type']}")
print(f"Size: {row['file_size']} bytes")

# Save file content to disk
if row['content_available']:
    from pathlib import Path
    file_path = Path(row['source']) / row['path']
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'wb') as f:
        f.write(row['content'])

# Filter by file type
images = dataset['train'].filter(lambda x: x['file_type'] == 'image')
print(f"Total images: {len(images)}")
```

## Dataset Creation

### Source Data

This dataset aggregates files from:
1. Epstein Estate release (September 2025)
2. Epstein Estate release (November 2025)  
3. House Committee on Oversight and Accountability - DOJ release (September 2025)

All materials were obtained through official channels including FOIA requests and public releases by government entities.

### Data Processing

Files were converted to Parquet format using PyArrow with:
- 500 MB maximum shard size (136 total shards)
- Snappy compression
- Binary content preservation
- File metadata extraction

**Note**: 2 files exceeding 2GB (PyArrow's binary value limit) are included with metadata only. Their `content` field is null and `content_available` is false.

## Considerations

- **Size**: 115+ GB total - ensure adequate storage and bandwidth
- **Content**: Contains legal documents, personal correspondence, and media
- **Large Files**: 2 files >2GB have metadata only (content not embedded)
- **Privacy**: Materials are from public releases; review applicable privacy considerations

## License

These materials are considered public domain as they are works of the U.S. Government or were publicly released through official channels. See [USA.gov Government Works](https://www.usa.gov/government-works) for more information.

Individual documents may be subject to additional restrictions or copyright claims. Users should review applicable laws and regulations for their jurisdiction.

## Citation

```bibtex
@dataset{epstractor2025,
  title={Epstractor: Epstein Archives Dataset},
  author={Various Government Sources},
  year={2025},
  publisher={Hugging Face},
  howpublished={\url{https://huggingface.co/datasets/bsmith925/epstractor}},
  note={Aggregated from Epstein Estate releases and DOJ FOIA materials}
}
```

## Maintenance

Dataset created: 2025-11-16  
Last updated: 2025-11-16  
Version: 1.0.0


# UPLOAD_WORKFLOW.md

# Complete Workflow: Upload to HuggingFace with Subsets

## Summary

Your dataset upload has been configured to use **subsets** instead of a single monolithic dataset. This allows users to:
- Load only the data they need
- Access different sources independently  
- Combine subsets as needed
- Stream large subsets without loading all into memory

## Files Created/Modified

### New Files
1. **`huggingface/upload_dataset.py`** - Upload script with subset support
2. **`huggingface/UPLOAD_GUIDE.md`** - Detailed guide
3. **`huggingface/SUBSET_UPLOAD_SUMMARY.md`** - Implementation summary
4. **`examples/load_subsets.py`** - 7 example patterns for users

### Modified Files
1. **`Makefile`** - Updated upload targets to use new script

## Quick Start: Upload Your Dataset

### Step 1: Set HuggingFace Token

```bash
# Set your token (get it from https://huggingface.co/settings/tokens)
export HF_TOKEN="hf_your_token_here"
```

### Step 2: Dry Run (Optional but Recommended)

```bash
# See what will be uploaded without uploading
uv run python -m huggingface.upload_dataset --dry-run
```

Expected output:
```
Dataset Upload Configuration
Repository: bsmith925/epstractor
Branch: main
Private: False

Subsets to upload:
  • epstein_estate_2025_09: huggingface/datasets/epstractor/data/epstein_estate_2025_09/*.parquet
  • epstein_estate_2025_11: huggingface/datasets/epstractor/data/epstein_estate_2025_11/*.parquet
  • house_doj_2025_09: huggingface/datasets/epstractor/data/house_doj_2025_09/*.parquet
```

### Step 3: Upload!

```bash
# Simple one-liner
make hf-upload

# Or directly with Python
uv run python -m huggingface.upload_dataset
```

This will:
1. ✅ Discover the 3 parquet subdirectories
2. ✅ Load them as named subsets
3. ✅ Validate schemas (7 columns each)
4. ✅ Push to `bsmith925/epstractor` on HuggingFace Hub

Expected time: 2-16 hours (depends on upload speed for 72GB)
Expected memory: 2-4 GB peak (mostly metadata)

## Memory Question - ANSWERED ✅

**Q: Do I have to worry about memory for uploading it to hub?**

**A: No!**

- ✅ Uses Git LFS: Uploads parquet files directly without loading into memory
- ✅ Metadata only: Only reads schemas, not actual data
- ✅ Peak memory: ~2-4 GB (for validation and buffers)
- ✅ Your 72GB stays on disk
- ✅ No risk of OOM errors

The main concern is **upload time** (hours), not memory (GB).

## Dataset Structure on Hub

```
bsmith925/epstractor
├── epstein_estate_2025_09/     # 5 files, 86 MB
├── epstein_estate_2025_11/     # 26,035 files, ~36 GB  
└── house_doj_2025_09/          # 33,380 files, ~36 GB
```

Total: 59,420 files, 72 GB, 3 subsets

## How Users Load It

### Basic Loading

```python
from datasets import load_dataset

# Load all subsets
ds = load_dataset("bsmith925/epstractor")
# Returns DatasetDict with 3 subsets

# Access individual subsets
epstein_09 = ds["epstein_estate_2025_09"]
epstein_11 = ds["epstein_estate_2025_11"]
house = ds["house_doj_2025_09"]
```

### Load Single Subset

```python
# Only load what you need
house = load_dataset("bsmith925/epstractor", split="house_doj_2025_09")
```

### Streaming (for large subsets)

```python
# Don't load all into memory
epstein_11 = load_dataset(
    "bsmith925/epstractor",
    split="epstein_estate_2025_11", 
    streaming=True
)

# Process in batches
for batch in epstein_11.iter(batch_size=100):
    process(batch)
```

## Benefits vs. Single Dataset

| Aspect | Old (Single) | New (Subsets) |
|--------|--------------|---------------|
| Load time | Load all 72GB | Load only what's needed |
| Memory usage | High for full dataset | Low with streaming |
| Source separation | Manual filtering | Built-in subsets |
| User flexibility | Limited | High (mix & match) |
| Train/eval splits | Manual creation | Use different subsets |
| Citation | Cite whole dataset | Cite specific subsets |

## Example Use Cases

See `examples/load_subsets.py` for 7 complete examples:

1. **Load all subsets** - Get everything as DatasetDict
2. **Load single subset** - Just House DOJ data
3. **Streaming** - Process 36GB without loading into memory
4. **Filtering** - Get only PDFs, or files >10MB
5. **Train/eval split** - Use different subsets for train/eval
6. **Combine subsets** - Merge all into one dataset
7. **Batch processing** - Stream and process in chunks

## Verification After Upload

```bash
# Check the repo
huggingface-cli list-repo bsmith925/epstractor --repo-type dataset

# Or visit
open https://huggingface.co/datasets/bsmith925/epstractor
```

Then test loading:

```python
from datasets import load_dataset

ds = load_dataset("bsmith925/epstractor")
print(ds)
# Should show all 3 subsets

# Quick sanity check
for name, dataset in ds.items():
    print(f"{name}: {len(dataset):,} rows")
```

## Troubleshooting

### Token Issues
```bash
# Check token is set
echo $HF_TOKEN

# Or use huggingface-cli
huggingface-cli login
```

### Upload Interrupted
Simply run the command again. Git LFS will resume from where it left off.

### Check Upload Progress
```bash
# List files in the repo
huggingface-cli list-repo bsmith925/epstractor --repo-type dataset
```

## Custom Upload Options

```bash
# Make dataset private
uv run python -m huggingface.upload_dataset --private

# Different repo
uv run python -m huggingface.upload_dataset --repo-id youruser/dataset-name

# Different branch
uv run python -m huggingface.upload_dataset --branch dev

# Custom data directory
uv run python -m huggingface.upload_dataset --data-dir path/to/data

# See all options
uv run python -m huggingface.upload_dataset --help
```

## Ready to Go! 🚀

Your dataset is configured and ready to upload with proper subsets. When you're ready:

```bash
export HF_TOKEN="your_token"
make hf-upload
```

Then share with the world:
```
https://huggingface.co/datasets/bsmith925/epstractor
```

Users can load it with:
```python
from datasets import load_dataset
ds = load_dataset("bsmith925/epstractor")
```

## Need Help?

- Upload guide: `huggingface/UPLOAD_GUIDE.md`
- Implementation details: `huggingface/SUBSET_UPLOAD_SUMMARY.md`
- User examples: `examples/load_subsets.py`
- HuggingFace docs: https://huggingface.co/docs/datasets/


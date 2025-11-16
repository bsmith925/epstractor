# Removing Splits: train → {source_name}

## Why?

Using `train-*.parquet` implies ML training data with train/test splits. For document archives where each source is a distinct collection, it's cleaner to remove the split concept entirely and name files after the source.

## Changes Made:

1. ✅ **Created rename script:** `huggingface/rename_to_no_split.py`
2. ✅ **Updated generator:** `huggingface/to_parquet.py` now outputs `{source_name}.parquet`
3. ✅ **Updated examples:** `examples/load_subsets.py` - removed all `split=` parameters
4. ✅ **Updated validator:** `huggingface/validate_dataset.py` - uses `name=` instead of `split=`

## How to Rename (Efficient Approach):

### Step 1: Dry Run (Preview Changes)
```bash
uv run python -m huggingface.rename_to_no_split --dry-run
```

### Step 2: Rename Local Files
```bash
uv run python -m huggingface.rename_to_no_split --local-only
```

This renames files in `huggingface/datasets/epstractor/data/`:
- `train-00000-of-00001.parquet` → `epstein_estate_2025_09-00000-of-00001.parquet`
- `train-00001-of-00075.parquet` → `epstein_estate_2025_11-00001-of-00075.parquet`
- etc.

### Step 3: Update HuggingFace Hub
```bash
export HF_TOKEN="your_token_here"
uv run python -m huggingface.rename_to_no_split
```

This will:
1. **Delete** old `train-*.parquet` files from Hub (API calls, fast)
2. **Upload** new `data-*.parquet` files to Hub
3. **Git LFS deduplication** - since content is identical, only filenames change!
   - No 73GB re-upload needed
   - Only metadata/references update

### Step 4: Verify
Visit: https://huggingface.co/datasets/bsmith925/epstractor

Check that files are now `data-*.parquet` and repo size hasn't doubled.

## Expected Timeline:

- **Local rename:** < 1 minute
- **Hub delete:** ~5-10 minutes (API calls for each file)
- **Hub upload:** ~10-30 minutes (Git LFS deduplication is fast)
- **Total:** ~15-40 minutes vs 12+ hours for full re-upload

## API Changes for Users:

### Before (train split):
```python
# Load single subset - requires split parameter
ds = load_dataset("bsmith925/epstractor", name="house_doj_2025_09", split="train")

# Load all, access split - nested structure
ds = load_dataset("bsmith925/epstractor")
house = ds["house_doj_2025_09"]["train"]  # Extra nesting level
```

### After (no splits):
```python
# Load single subset - returns Dataset directly!
ds = load_dataset("bsmith925/epstractor", name="house_doj_2025_09")

# Load all - direct access, no nesting!
ds = load_dataset("bsmith925/epstractor")
house = ds["house_doj_2025_09"]  # Clean and simple
```

## Safety Notes:

- ✅ Script has `--dry-run` mode to preview changes
- ✅ Git LFS ensures no duplicate uploads (checks content hash)
- ✅ Original files preserved until confirmed working
- ✅ Can run `--local-only` first to test locally

## Rollback (if needed):

If something goes wrong, you can rollback:

```bash
# Locally
cd huggingface/datasets/epstractor/data
find . -name 'data-*.parquet' -exec sh -c 'mv "$1" "${1/data-/train-}"' _ {} \;

# On Hub
# Re-run original upload_dataset.py script
uv run python -m huggingface.upload_dataset
```

## Questions?

The script supports:
- `--dry-run` - Preview without changes
- `--local-only` - Only rename local files
- `--hub-only` - Only update Hub (assumes local already renamed)
- `--log-level DEBUG` - Verbose output

```bash
uv run python -m huggingface.rename_to_no_split --help
```


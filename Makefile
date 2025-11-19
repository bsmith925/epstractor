# Makefile for dataset management and Hugging Face operations

# ============================================================================
# Configuration Variables
# ============================================================================
# Hugging Face settings
HF_CONFIG ?= epstein_all
HF_CFG_DIR ?= huggingface/datasets/configs

# Download settings
DATASET ?= epstein_estate_2025_11
CONFIG_DIR ?= downloads/configs
OUTPUT_DIR ?= downloads/datasets
MAX_DRIVE ?= 4
MAX_HTTP ?= 8

# Python runner
PY ?= uv run python

# ============================================================================
# Hugging Face Operations
# ============================================================================
# Valid actions for HF operations
ACTIONS := parquet

# Reserved targets that shouldn't be treated as configs
RESERVED := dedupe download-epstein help hf-upload hf-update-readme upload-dataset validate-dataset

# Parse command-line goals
CONFIG_GOAL := $(firstword $(filter-out $(ACTIONS) $(RESERVED),$(MAKECMDGOALS)))
ACTION_GOAL := $(firstword $(filter $(ACTIONS),$(MAKECMDGOALS)))

# Set defaults if not specified
ifeq ($(CONFIG_GOAL),)
  CONFIG_GOAL := $(HF_CONFIG)
endif
ifeq ($(ACTION_GOAL),)
  ACTION_GOAL := parquet
endif

.PHONY: do $(ACTIONS)

# Main routing target for HF operations
do:
	@ACTUAL_TARGET="$(filter-out $(ACTIONS) do,$(MAKECMDGOALS))"; \
	if echo "$(RESERVED)" | grep -qw "$$ACTUAL_TARGET"; then \
		exit 0; \
	elif echo "$(RESERVED)" | grep -qw "$(CONFIG_GOAL)"; then \
		exit 0; \
	elif [ -d "downloads/datasets/$(CONFIG_GOAL)" ]; then \
		exit 0; \
	elif [ "$(ACTION_GOAL)" = "parquet" ]; then \
		$(PY) -u -m huggingface.to_parquet $(CONFIG_GOAL) --configs-dir $(HF_CFG_DIR); \
	elif [ "$(ACTION_GOAL)" = "upload-dataset" ]; then \
		$(PY) -m huggingface.upload_dataset; \
	fi

# Swallow action goals so they don't trigger additional rules
$(ACTIONS):
	@:

# ============================================================================
# Dataset Upload
# ============================================================================
.PHONY: upload-dataset
upload-dataset:
	$(PY) -m huggingface.upload_dataset

# ============================================================================
# Dataset Validation
# ============================================================================
.PHONY: validate-dataset
validate-dataset:
	$(PY) -m huggingface.validate_dataset

# ============================================================================
# Update README on HuggingFace Hub
# ============================================================================
.PHONY: hf-update-readme
hf-update-readme:
	$(PY) -m huggingface.update_readme

# ============================================================================
# Dataset Download
# ============================================================================
DEFAULT_ARGS = --config-dir $(CONFIG_DIR) --output-dir $(OUTPUT_DIR) --source $(DATASET) --max-drive-workers $(MAX_DRIVE) --max-http-workers $(MAX_HTTP)

.PHONY: download-epstein
download-epstein:
	$(eval DOWNLOAD_ARGS := $(wordlist 2,$(words $(MAKECMDGOALS)),$(MAKECMDGOALS)))
	@echo "tmux session: epstein_dl"
	@echo "Download args: $(DEFAULT_ARGS) $(DOWNLOAD_ARGS)"
	@if tmux has-session -t epstein_dl 2>/dev/null; then \
		echo "Reusing existing tmux session"; \
		tmux set-option -t epstein_dl:0 remain-on-exit on >/dev/null; \
		tmux send-keys -t epstein_dl "cd $(CURDIR) && uv run download-datasets $(DEFAULT_ARGS) $(DOWNLOAD_ARGS)" C-m; \
	else \
		tmux new-session -d -s epstein_dl "cd $(CURDIR) && uv run download-datasets $(DEFAULT_ARGS) $(DOWNLOAD_ARGS)"; \
		tmux set-option -t epstein_dl:0 remain-on-exit on >/dev/null; \
	fi
	@tmux attach -t epstein_dl

# ============================================================================
# Dataset Deduplication
# ============================================================================
.PHONY: dedupe
dedupe:
	@DATASETS="$(filter-out dedupe,$(MAKECMDGOALS))"; \
	if [ -z "$$DATASETS" ]; then \
		DATASETS="$(DATASET)"; \
	fi; \
	if [ -z "$$DATASETS" ]; then \
		echo "Error: No dataset specified. Usage: make dedupe <dataset1> [dataset2 ...]"; \
		echo "Example: make dedupe house_doj_2025_09"; \
		echo "Example: make dedupe epstein_estate_2025_09 epstein_estate_2025_11 house_doj_2025_09"; \
		exit 1; \
	fi; \
	for ds in $$DATASETS; do \
		DATASET_DIR="$(OUTPUT_DIR)/$$ds"; \
		if [ ! -d "$$DATASET_DIR" ]; then \
			echo "Error: Dataset directory not found: $$DATASET_DIR"; \
			exit 1; \
		fi; \
		echo "Deduplicating manifest for dataset: $$ds"; \
		$(PY) downloads/dedupe.py --dataset-dir "$$DATASET_DIR"; \
		echo ""; \
	done

# ============================================================================
# Simplified Upload Target
# ============================================================================
.PHONY: hf-upload
hf-upload:
	@echo "Uploading parquet dataset to HuggingFace Hub with subsets..."
	$(PY) -m huggingface.upload_dataset

# ============================================================================
# Help
# ============================================================================
.PHONY: help
help:
	@echo "Dataset Management Makefile"
	@echo ""
	@echo "Parquet Workflow:"
	@echo "  make <config> parquet                       - Convert datasets to Parquet"
	@echo "  make <config> upload-dataset                - Upload to HuggingFace Hub"
	@echo ""
	@echo "Quick Upload (Recommended):"
	@echo "  make hf-upload                              - Upload with subsets to HF Hub"
	@echo "  Example: export HF_TOKEN=hf_xxx && make hf-upload"
	@echo ""
	@echo "Full Workflow Examples:"
	@echo "  make epstein_all parquet"
	@echo "  make epstein_all upload-dataset"
	@echo "  make hf-upload"
	@echo ""
	@echo "Dataset Operations:"
	@echo "  make download-epstein [args]                - Download datasets in tmux"
	@echo "  make dedupe <dataset1> [dataset2 ...]       - Deduplicate manifests"
	@echo ""
	@echo "Configuration:"
	@echo "  HF_CONFIG=$(HF_CONFIG)  (YAML config in $(HF_CFG_DIR)/)"
	@echo "  repo_id set in: $(HF_CFG_DIR)/$(HF_CONFIG).yaml"
	@echo "  DATASET=$(DATASET)"
	@echo "  OUTPUT_DIR=$(OUTPUT_DIR)"

# ============================================================================
# Catch-all rule (must be last)
# ============================================================================
# Route unknown goals to HF operations or consume as arguments
%: do
	@:

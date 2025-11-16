# huggingface/validate_dataset.py

"""
Validate dataset on HuggingFace Hub using streaming mode (no download).

Checks:
- Dataset loads successfully
- Expected subsets exist
- Schema is correct
- No unwanted files (e.g., dedupe_report.json)
- Binary content is usable (test OCR on PDFs)
- Basic statistics from sampling
"""

from __future__ import annotations

import argparse
import io
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from rich.console import Console
from rich.table import Table


def validate_dataset(
    repo_id: str,
    expected_subsets: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    sample_size: int = 50,
    test_binary: bool = True,
) -> dict:
    """
    Validate dataset on HuggingFace Hub using streaming (no full download).
    
    Args:
        repo_id: HuggingFace dataset repository ID
        expected_subsets: List of expected subset names
        exclude_patterns: Patterns that should NOT appear in file paths
        sample_size: Number of rows to sample for validation
        test_binary: Whether to test binary content (PDF extraction, etc.)
        
    Returns:
        Validation results dictionary
    """
    console = Console()
    exclude_patterns = exclude_patterns or ["dedupe_report.json"]
    
    console.print(f"\n[bold cyan]Validating Dataset: {repo_id}[/bold cyan]")
    console.print(f"[dim]Using streaming mode - sampling {sample_size} rows per subset (large files may take time)[/dim]\n")
    
    results = {
        "repo_id": repo_id,
        "success": True,
        "errors": [],
        "warnings": [],
        "stats": {},
    }
    
    # 1. Load dataset in streaming mode
    console.print("[cyan]Loading dataset info from Hub...[/cyan]")
    try:
        # Load each subset in streaming mode
        dataset = {}
        if expected_subsets:
            for subset_name in expected_subsets:
                console.print(f"  Loading {subset_name} (streaming)...")
                dataset[subset_name] = load_dataset(
                    repo_id, 
                    name=subset_name, 
                    streaming=True
                )
        else:
            results["errors"].append("No expected subsets provided")
            return results
        console.print(f"[green]✓ Dataset loaded successfully[/green]\n")
    except Exception as e:
        results["success"] = False
        results["errors"].append(f"Failed to load dataset: {e}")
        console.print(f"[red]✗ Failed to load dataset: {e}[/red]")
        return results
    
    # 2. Validate subsets by sampling
    console.print("[bold]Validating Subsets:[/bold]")
    available_subsets = list(dataset.keys())
    
    subset_table = Table()
    subset_table.add_column("Subset", style="cyan")
    subset_table.add_column("Sample Size", justify="right")
    subset_table.add_column("Status", style="green")
    
    for subset_name in available_subsets:
        subset_table.add_row(subset_name, f"{sample_size}", "streaming")
    
    console.print(subset_table)
    console.print()
    
    # Check if expected subsets exist
    if expected_subsets:
        missing = set(expected_subsets) - set(available_subsets)
        if missing:
            results["warnings"].append(f"Missing expected subsets: {missing}")
            console.print(f"[yellow]⚠ Missing expected subsets: {missing}[/yellow]")
    
    # 3. Sample and validate each subset
    console.print("[bold]Sampling and Validating Data:[/bold]\n")
    expected_columns = {
        "path": str,
        "source": str,
        "file_type": str,
        "file_size": int,
        "extension": str,
        "content": bytes,
        "content_available": bool,
    }
    
    for subset_name in available_subsets:
        console.print(f"[cyan]{subset_name}[/cyan]")
        subset = dataset[subset_name]
        
        # Sample rows (with progress) - only take smaller files for speed
        console.print(f"  [dim]Sampling {sample_size} rows (skipping files >100MB for speed)...[/dim]")
        max_file_size = 100 * 1024 * 1024  # 10MB limit for sampling (much stricter)
        samples = []
        skipped_large = 0
        checked = 0
        
        # First pass: collect metadata WITHOUT downloading content
        for sample in subset:
            checked += 1
            
            # Skip very large files to speed up sampling (check size before downloading content)
            if sample["file_size"] > max_file_size:
                skipped_large += 1
                # Don't download content for large files
                if checked % 100 == 0:
                    console.print(f"  [dim]Checked {checked} files, sampled {len(samples)}/{sample_size} (skipped {skipped_large} large)...[/dim]", end="\r")
                continue
            
            samples.append(sample)
            if len(samples) >= sample_size:
                break
            
            if checked % 20 == 0:
                console.print(f"  [dim]Checked {checked} files, sampled {len(samples)}/{sample_size} (skipped {skipped_large} large)...[/dim]", end="\r")
            
            # Stop if we've checked too many files without getting enough samples
            if checked > sample_size * 20:
                console.print(f"\n  [yellow]⚠ Stopping early after checking {checked} files, only found {len(samples)} small files[/yellow]")
                break
        
        if not samples:
            results["errors"].append(f"{subset_name}: No data found")
            console.print(f"  [red]✗ No data found[/red]")
            continue
        
        # Check schema from first sample
        first_sample = samples[0]
        actual_columns = set(first_sample.keys())
        expected_col_names = set(expected_columns.keys())
        
        missing_cols = expected_col_names - actual_columns
        extra_cols = actual_columns - expected_col_names
        
        if missing_cols:
            error = f"{subset_name}: missing columns {missing_cols}"
            results["errors"].append(error)
            console.print(f"  [red]✗ {error}[/red]")
            continue
        if extra_cols:
            warning = f"{subset_name}: unexpected columns {extra_cols}"
            results["warnings"].append(warning)
            console.print(f"  [yellow]⚠ {warning}[/yellow]")
        
        console.print(f"  [green]✓ Schema correct[/green]")
        
        # Check for unwanted files
        unwanted_count = sum(
            1 for sample in samples 
            if any(pattern in sample["path"] for pattern in exclude_patterns)
        )
        
        if unwanted_count > 0:
            error = f"{subset_name}: Found {unwanted_count} files matching exclude patterns"
            results["errors"].append(error)
            console.print(f"  [red]✗ {error}[/red]")
            # Show examples
            for sample in samples:
                if any(pattern in sample["path"] for pattern in exclude_patterns):
                    console.print(f"    [dim]{sample['path']}[/dim]")
                    break
        else:
            console.print(f"  [green]✓ No unwanted files found[/green]")
        
        # Statistics from sample
        file_types = Counter(s["file_type"] for s in samples)
        extensions = Counter(s["extension"] for s in samples)
        content_available = sum(1 for s in samples if s["content_available"])
        
        console.print(f"  [dim]Checked {checked} files, sampled {len(samples)}, skipped {skipped_large} large (>10MB)[/dim]")
        console.print(f"  [dim]File types: {', '.join(f'{ft}:{ct}' for ft, ct in file_types.most_common(5))}[/dim]")
        console.print(f"  [dim]Top extensions: {', '.join(f'{ext}:{ct}' for ext, ct in extensions.most_common(5))}[/dim]")
        console.print(f"  [dim]Content available: {content_available}/{len(samples)} ({100*content_available/len(samples):.1f}%)[/dim]")
        
        results["stats"][subset_name] = {
            "sampled": len(samples),
            "file_types": dict(file_types),
            "extensions": dict(extensions),
            "content_available": content_available,
        }
        
        console.print()
    
    # 4. Test binary content usability
    if test_binary:
        console.print("[bold]Testing Binary Content:[/bold]")
        test_pdfs = []
        test_images = []
        
        for subset_name in available_subsets:
            subset = dataset[subset_name]
            for sample in subset.take(sample_size):
                if sample["extension"] == ".pdf" and sample["content_available"] and sample["content"]:
                    test_pdfs.append((subset_name, sample))
                    if len(test_pdfs) >= 3:
                        break
                elif sample["extension"] in [".jpg", ".jpeg", ".png"] and sample["content_available"] and sample["content"]:
                    test_images.append((subset_name, sample))
                    if len(test_images) >= 3:
                        break
                if test_pdfs and test_images:
                    break
        
        # Test PDF content
        if test_pdfs:
            try:
                import pypdf
                pdf_tested = 0
                for subset_name, sample in test_pdfs:
                    try:
                        pdf_bytes = io.BytesIO(sample["content"])
                        reader = pypdf.PdfReader(pdf_bytes)
                        num_pages = len(reader.pages)
                        first_page_text = reader.pages[0].extract_text()
                        pdf_tested += 1
                        console.print(f"  [green]✓ PDF readable: {sample['path'][:50]}... ({num_pages} pages)[/green]")
                    except Exception as e:
                        results["warnings"].append(f"PDF read error: {sample['path']}: {e}")
                        console.print(f"  [yellow]⚠ PDF error: {sample['path'][:50]}...: {e}[/yellow]")
                
                if pdf_tested > 0:
                    console.print(f"  [green]✓ Successfully tested {pdf_tested} PDF(s)[/green]")
            except ImportError:
                console.print(f"  [yellow]⚠ pypdf not installed, skipping PDF test[/yellow]")
        
        # Test image content
        if test_images:
            try:
                from PIL import Image
                img_tested = 0
                for subset_name, sample in test_images:
                    try:
                        img_bytes = io.BytesIO(sample["content"])
                        img = Image.open(img_bytes)
                        console.print(f"  [green]✓ Image readable: {sample['path'][:50]}... ({img.size[0]}x{img.size[1]})[/green]")
                        img_tested += 1
                    except Exception as e:
                        results["warnings"].append(f"Image read error: {sample['path']}: {e}")
                        console.print(f"  [yellow]⚠ Image error: {sample['path'][:50]}...: {e}[/yellow]")
                
                if img_tested > 0:
                    console.print(f"  [green]✓ Successfully tested {img_tested} image(s)[/green]")
            except ImportError:
                console.print(f"  [yellow]⚠ PIL not installed, skipping image test[/yellow]")
        
        console.print()
    
    # 5. Final verdict
    console.print("[bold]Validation Summary:[/bold]")
    if results["errors"]:
        results["success"] = False
        console.print(f"[bold red]✗ Validation FAILED with {len(results['errors'])} error(s)[/bold red]")
        for error in results["errors"]:
            console.print(f"  [red]• {error}[/red]")
    elif results["warnings"]:
        console.print(f"[bold yellow]⚠ Validation passed with {len(results['warnings'])} warning(s)[/bold yellow]")
        for warning in results["warnings"]:
            console.print(f"  [yellow]• {warning}[/yellow]")
    else:
        console.print("[bold green]✓ Validation PASSED - Dataset looks good![/bold green]")
    
    console.print()
    
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate dataset on HuggingFace Hub using streaming (no full download)"
    )
    parser.add_argument(
        "--repo-id",
        default="bsmith925/epstractor",
        help="HuggingFace dataset repository ID (default: bsmith925/epstractor)",
    )
    parser.add_argument(
        "--expected-subsets",
        nargs="+",
        default=["epstein_estate_2025_09", "epstein_estate_2025_11", "house_doj_2025_09"],
        help="Expected subset names",
    )
    parser.add_argument(
        "--exclude-patterns",
        nargs="+",
        default=["dedupe_report.json"],
        help="Patterns that should not appear in file paths",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of rows to sample per subset (default: 50)",
    )
    parser.add_argument(
        "--no-test-binary",
        action="store_true",
        help="Skip binary content testing (PDF, images)",
    )
    
    args = parser.parse_args()
    
    results = validate_dataset(
        repo_id=args.repo_id,
        expected_subsets=args.expected_subsets,
        exclude_patterns=args.exclude_patterns,
        sample_size=args.sample_size,
        test_binary=not args.no_test_binary,
    )
    
    return 0 if results["success"] else 1


if __name__ == "__main__":
    import sys
    sys.exit(main())


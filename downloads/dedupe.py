# downloads/dedupe.py

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


def _load_manifest(path: Path) -> List[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"ERROR: Failed to read manifest {path}: {exc}", file=sys.stderr)
        raise


def _write_manifest(path: Path, manifest: List[dict]) -> None:
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def dedupe_manifest(
    dataset_dir: Path,
    dry_run: bool = False,
    backup: bool = True,
    report_file: Optional[Path] = None,
) -> Dict:
    """
    Dedupe identical entries in .manifest.json by (path, md5).
    - Keeps the first occurrence of each unique (path, md5) pair, preserving order.
    - Skips subsequent entries with the same (path, md5).
    - Does NOT resolve naming conflicts (same path with different md5); these are reported.
    """
    manifest_path = dataset_dir / ".manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    entries = _load_manifest(manifest_path)

    seen_keys: Set[Tuple[str, Optional[str]]] = set()
    md5s_by_path: Dict[str, Set[str]] = {}
    deduped: List[dict] = []
    duplicates_removed = 0
    md5_missing = 0

    for entry in entries:
        rel_path = entry.get("path")
        md5 = entry.get("md5")

        if md5 is None:
            # If md5 is missing, do not attempt to dedupe; preserve as-is
            md5_missing += 1
            deduped.append(entry)
            continue

        key = (rel_path, md5)
        if key in seen_keys:
            duplicates_removed += 1
            continue

        seen_keys.add(key)
        if rel_path is not None:
            md5s_by_path.setdefault(rel_path, set()).add(md5)
        deduped.append(entry)

    conflict_paths = {p: sorted(list(md5s)) for p, md5s in md5s_by_path.items() if len(md5s) > 1}

    stats = {
        "dataset_dir": str(dataset_dir),
        "entries_before": len(entries),
        "entries_after": len(deduped),
        "duplicates_removed": duplicates_removed,
        "md5_missing_entries": md5_missing,
        "conflicting_paths": len(conflict_paths),
    }

    # Write outputs
    if not dry_run:
        if backup and duplicates_removed:
            timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
            backup_path = manifest_path.with_suffix(f".json.bak.{timestamp}")
            backup_path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        if duplicates_removed:
            _write_manifest(manifest_path, deduped)

    # Report
    report = {
        "stats": stats,
        "examples": {
            "conflicts_sample": [
                {"path": p, "md5s": conflict_paths[p]}
                for p in list(conflict_paths.keys())[:20]
            ]
        },
    }
    if report_file is None:
        report_file = dataset_dir / "dedupe_report.json"
    try:
        report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    except Exception:
        # Non-fatal; still return stats
        pass

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dedupe identical entries in a dataset .manifest.json by path+md5."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Path to a dataset directory containing .manifest.json.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and report dedupe without writing changes.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not write a backup manifest when modifying.",
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        help="Optional path to write a JSON report (default: <dataset-dir>/dedupe_report.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = dedupe_manifest(
        dataset_dir=args.dataset_dir,
        dry_run=args.dry_run,
        backup=not args.no_backup,
        report_file=args.report_file,
    )
    stats = report.get("stats", {})
    print(
        f"Dedupe complete for {stats.get('dataset_dir')}: "
        f"{stats.get('entries_before')} -> {stats.get('entries_after')} "
        f"(removed {stats.get('duplicates_removed')} duplicates). "
        f"md5-missing entries: {stats.get('md5_missing_entries')}. "
        f"conflicting paths: {stats.get('conflicting_paths')}."
    )
    conflicts = report.get("examples", {}).get("conflicts_sample", [])
    if conflicts:
        print("Example conflicting paths (different md5 for same path):")
        for c in conflicts[:10]:
            print(f" - {c['path']} ({', '.join(c['md5s'])})")


if __name__ == "__main__":
    main()



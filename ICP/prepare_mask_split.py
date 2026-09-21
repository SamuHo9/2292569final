"""Prepare pre-ICP, patient-level mask splits from the audited split manifest.

The current project split was originally materialized after SPHARM. This helper
reuses that exact patient assignment and the ALL-side status tables to stage the
original hippocampus masks before ICP. It copies inputs; it never moves or
deletes source data. Only subjects with an audited side label are staged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_MANIFEST = PROJECT_ROOT / "SPHARM" / "split_data" / "current_split_manifest.json"
DEFAULT_LEFT_STATUS = PROJECT_ROOT / "SPHARM" / "split_data" / "ALL_Left_file_status.csv"
DEFAULT_RIGHT_STATUS = PROJECT_ROOT / "SPHARM" / "split_data" / "ALL_Right_file_status.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "ICP" / "fixed_reference_rerun"
VOLUME_SUFFIXES = (".nii.gz", ".nii", ".nrrd", ".mgz", ".hdr", ".img")
SUBJECT_RE = re.compile(r"(?i)(?<![A-Za-z0-9])(?P<subject>sub-[A-Za-z0-9]+)(?![A-Za-z0-9])")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strip_volume_suffix(name: str) -> str:
    lower = name.lower()
    for suffix in VOLUME_SUFFIXES:
        if lower.endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def subject_id_from_name(name: str) -> str:
    matches = list(SUBJECT_RE.finditer(name))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one sub-* ID in filename: {name}")
    return matches[0].group("subject")


def normalize_side_label(value: object) -> str:
    label = str(value or "").strip().lower()
    if label not in {"healthy", "tle"}:
        raise ValueError(f"Expected side label Healthy or TLE; got {value!r}")
    return "Healthy" if label == "healthy" else "TLE"


def infer_filename_side(name: str) -> str | None:
    stem = strip_volume_suffix(name).lower()
    left = bool(re.search(r"(?:^|[_-])(?:left|lh)(?:[_-]|$)", stem))
    right = bool(re.search(r"(?:^|[_-])(?:right|rh)(?:[_-]|$)", stem))
    if left and right:
        raise ValueError(f"Filename contains both left and right markers: {name}")
    if left:
        return "left"
    if right:
        return "right"
    return None


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_assignments(path: Path) -> Dict[str, dict]:
    manifest = read_json(path)
    if manifest.get("policy") != "patient_subject_level_stratified_split":
        raise ValueError(f"Unsupported split policy in {path}: {manifest.get('policy')!r}")
    assignments: Dict[str, dict] = {}
    for row in manifest.get("subjects", []):
        sid = str(row.get("subject_id", "")).strip()
        split = row.get("split")
        if not sid or split not in {"train", "test"} or sid in assignments:
            raise ValueError(f"Invalid or duplicate subject assignment: {row}")
        assignments[sid] = row
    if not assignments:
        raise ValueError(f"No subject assignments in {path}")
    return assignments


def load_side_status(path: Path, side: str, assignments: Mapping[str, dict]) -> Dict[str, dict]:
    expected_side_label = f"{side}_label"
    records: Dict[str, dict] = {}
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"FileName", "Status", "Dataset"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{path} must contain columns {sorted(required)}")
        for row in reader:
            filename = str(row.get("FileName", "")).strip()
            sid = subject_id_from_name(filename)
            label = normalize_side_label(row.get("Status"))
            dataset = str(row.get("Dataset", "")).strip()
            if dataset not in {"Ds004469", "Ds005602"}:
                raise ValueError(f"Unexpected dataset {dataset!r} for {filename}")
            if sid not in assignments:
                raise ValueError(f"Status table subject absent from split manifest: {sid}")
            if sid in records:
                raise ValueError(f"Duplicate {side} status row for {sid}")
            manifest_label = assignments[sid].get(expected_side_label)
            if normalize_side_label(manifest_label) != label:
                raise ValueError(
                    f"{side} label mismatch for {sid}: split manifest={manifest_label!r}, status={label!r}"
                )
            records[sid] = {
                "subject_id": sid,
                "split": assignments[sid]["split"],
                "label": label,
                "binary_class": int(label == "TLE"),
                "dataset": dataset,
                "status_filename": filename,
            }
    if not records:
        raise ValueError(f"No {side} rows in status table: {path}")
    return records


def find_mask_files(source_dir: Path) -> List[Path]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Mask source directory not found: {source_dir}")
    files = []
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        lower = path.name.lower()
        if not any(lower.endswith(suffix) for suffix in VOLUME_SUFFIXES):
            continue
        if "hippocampus" not in lower or "intensity" in lower:
            continue
        files.append(path.resolve())
    return sorted(files, key=lambda item: item.name.lower())


def match_source_masks(source_dir: Path, side: str, status: Mapping[str, dict],
                       assignments: Mapping[str, dict]) -> Tuple[Dict[str, Path], List[dict]]:
    expected_ids = set(status)
    manifest_ids = set(assignments)
    found: Dict[str, Path] = {}
    excluded: List[dict] = []
    unknown: List[str] = []
    for path in find_mask_files(source_dir):
        sid = subject_id_from_name(path.name)
        filename_side = infer_filename_side(path.name)
        if filename_side is not None and filename_side != side:
            raise ValueError(f"Wrong-side mask in {source_dir}: {path.name}")
        if sid not in manifest_ids:
            unknown.append(path.name)
            continue
        if sid not in expected_ids:
            excluded.append({"subject_id": sid, "filename": path.name,
                             "reason": "subject has no audited status row for this side"})
            continue
        if sid in found:
            raise ValueError(f"Multiple {side} mask files for {sid}: {found[sid].name}, {path.name}")
        found[sid] = path
    if unknown:
        raise ValueError("Mask files contain IDs absent from the split manifest: " + ", ".join(unknown[:10]))
    missing = sorted(expected_ids - set(found))
    if missing:
        raise FileNotFoundError(f"Missing {side} hippocampus masks for {len(missing)} audited subjects: {missing[:10]}")
    return found, excluded


def write_labels_csv(path: Path, records: Mapping[str, dict], source_files: Mapping[str, Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Subject", "Group", "Class", "BinaryClass", "Dataset", "PatientID", "Split"])
        for sid, record in sorted(records.items()):
            mask_stem = strip_volume_suffix(source_files[sid].name)
            coefficient_subject = mask_stem if mask_stem.lower().endswith("_aligned") else mask_stem + "_aligned"
            binary = int(record["binary_class"])
            group = "Ipsilateral TLE (Diseased)" if binary else "Non-affected/control"
            writer.writerow([coefficient_subject, group, binary, binary,
                             record["dataset"], sid, record["split"]])


def prepare(left_source: Path, right_source: Path, manifest_path: Path,
            left_status_path: Path, right_status_path: Path, output_root: Path) -> dict:
    assignments = load_assignments(manifest_path)
    side_inputs = {
        "left": (left_source.resolve(), left_status_path.resolve()),
        "right": (right_source.resolve(), right_status_path.resolve()),
    }
    side_records = {
        side: load_side_status(status_path, side, assignments)
        for side, (_, status_path) in side_inputs.items()
    }
    matched = {}
    excluded = {}
    for side, (source_dir, _) in side_inputs.items():
        matched[side], excluded[side] = match_source_masks(
            source_dir, side, side_records[side], assignments
        )

    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root is not empty; choose a new path: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    side_summaries = {}
    file_records = []
    for side in ("left", "right"):
        summary = {"train": 0, "test": 0, "excluded_unstatused": len(excluded[side])}
        for split in ("train", "test"):
            destination = output_root / "masks" / side / split
            destination.mkdir(parents=True, exist_ok=True)
            split_records = {
                sid: row for sid, row in side_records[side].items() if row["split"] == split
            }
            for sid, row in sorted(split_records.items()):
                source = matched[side][sid]
                target = destination / source.name
                shutil.copy2(source, target)
                summary[split] += 1
                file_records.append({
                    **row,
                    "side": side,
                    "source": str(source),
                    "source_sha256": sha256(source),
                    "staged": str(target.relative_to(output_root)),
                })
            write_labels_csv(output_root / "labels" / side / f"{split}_labels.csv",
                             split_records, matched[side])
        side_summaries[side] = summary

    result = {
        "schema": "pre_icp_patient_split_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy": "reused_existing_patient_level_train_test_assignments",
        "source_split_manifest": str(manifest_path.resolve()),
        "source_split_manifest_sha256": sha256(manifest_path.resolve()),
        "status_tables": {
            "left": {"path": str(left_status_path.resolve()), "sha256": sha256(left_status_path.resolve())},
            "right": {"path": str(right_status_path.resolve()), "sha256": sha256(right_status_path.resolve())},
        },
        "sources": {side: str(source) for side, (source, _) in side_inputs.items()},
        "output_root": str(output_root),
        "assignments_total": len(assignments),
        "side_counts": side_summaries,
        "excluded_masks": excluded,
        "masks": file_records,
    }
    report_path = output_root / "pre_icp_split_manifest.json"
    report_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "masks"}, indent=2, ensure_ascii=False))
    print(f"Prepared input manifest: {report_path}")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-source", type=Path, required=True,
                        help="Extracted left hippocampus mask folder")
    parser.add_argument("--right-source", type=Path, required=True,
                        help="Extracted right hippocampus mask folder")
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--left-status", type=Path, default=DEFAULT_LEFT_STATUS)
    parser.add_argument("--right-status", type=Path, default=DEFAULT_RIGHT_STATUS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prepare(args.left_source, args.right_source, args.split_manifest,
            args.left_status, args.right_status, args.output_root)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

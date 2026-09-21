"""Extract the current SPHARM split and format it for the leakage-free runner.

This keeps each split separate, records explicit labels and provenance, and
creates a fresh data_root that Model/optuna_leakage_free_all.py can read.
The Optuna runner performs PLS-DA augmentation inside each training fold; this
script does not pre-augment or modify the test set.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path


COHORTS = ("All_coef", "All_coef_Ds004469", "All_coef_Ds005602")
SIDES = ("left", "right")
SPLITS = ("train", "test")
META = ["Subject", "Group", "Class", "BinaryClass", "DataType"]
FEATURE_RE = re.compile(r"^Coef_(\d+)$")
SUBJECT_RE = re.compile(r"(?i)(?:^|[_-])(sub[-_]?[a-z0-9]+)(?:_|$)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def patient_ids(subjects: list[str]) -> set[str]:
    result = set()
    for subject in subjects:
        match = SUBJECT_RE.search(subject)
        result.add(match.group(1).replace("_", "-").lower() if match else subject.lower())
    return result


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split-root", type=Path,
        default=project / "SPHARM" / "split_data" / "current_spharm_80_20_20260922",
        help="Current coefficient split containing All_coef*/{left,right}/{train,test}",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=project / "Model" / "current_spharm_80_20_20260922",
        help="New data_root for Model training; must not already exist",
    )
    parser.add_argument(
        "--python", type=Path, default=Path(sys.executable),
        help="Python interpreter that can run extract_ml_features_coef.py",
    )
    args = parser.parse_args()
    split_root = args.split_root.resolve()
    output_root = args.output_root.resolve()
    staging_root = output_root.with_name(output_root.name + ".building")
    extractor = project / "Data_Processing" / "extract_ml_features_coef.py"
    sample_manifest_path = split_root / "sample_manifest.csv"

    if not split_root.is_dir():
        raise FileNotFoundError(f"Split dataset not found: {split_root}")
    if output_root.exists() or staging_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite output/staging directory: {output_root} / {staging_root}"
        )
    if not extractor.is_file() or not sample_manifest_path.is_file():
        raise FileNotFoundError("Missing coefficient extractor or split sample manifest")

    _, split_rows = read_rows(sample_manifest_path)
    expected_counts = {}
    for row in split_rows:
        key = (row["DatasetVariant"], row["Side"], row["Split"])
        expected_counts[key] = expected_counts.get(key, 0) + 1

    staging_root.mkdir(parents=True)
    records = []
    group_ids: dict[tuple[str, str, str], set[str]] = {}
    contracts: dict[tuple[str, str, str], dict] = {}
    try:
        for cohort in COHORTS:
            for side in SIDES:
                train_subjects = []
                test_subjects = []
                train_contract = None
                test_contract = None
                for split in SPLITS:
                    source_dir = split_root / cohort / side / split
                    labels_csv = source_dir / "labels.csv"
                    if not source_dir.is_dir() or not labels_csv.is_file():
                        raise FileNotFoundError(f"Missing input directory or labels: {source_dir}")

                    command = [
                        str(args.python.resolve()), str(extractor),
                        "--spharm_dir", str(source_dir),
                        "--labels_csv", str(labels_csv),
                        "--coef_variant", "ellalign",
                    ]
                    completed = subprocess.run(
                        command, cwd=str(project / "Data_Processing"),
                        capture_output=True, text=True, encoding="utf-8", errors="replace",
                    )
                    if completed.stdout:
                        print(completed.stdout, end="", flush=True)
                    if completed.returncode:
                        if completed.stderr:
                            print(completed.stderr, file=sys.stderr, end="", flush=True)
                        raise RuntimeError(f"Feature extraction failed for {cohort}/{side}/{split}")

                    feature_csv = source_dir.parent / "ml_features" / f"{split}_coef_features.csv"
                    extractor_manifest = Path(str(feature_csv) + ".json")
                    if not feature_csv.is_file() or not extractor_manifest.is_file():
                        raise FileNotFoundError(f"Extractor output missing: {feature_csv}")
                    payload = json.loads(extractor_manifest.read_text(encoding="utf-8-sig"))
                    feature_contract = payload.get("feature_contract")
                    if not isinstance(feature_contract, dict) or feature_contract.get("coefficient_variant") != "ellalign":
                        raise ValueError(f"Expected ellalign provenance contract: {extractor_manifest}")

                    source_columns, rows = read_rows(feature_csv)
                    feature_columns = sorted(
                        (name for name in source_columns if FEATURE_RE.match(name)),
                        key=lambda name: int(FEATURE_RE.match(name).group(1)),
                    )
                    if len(feature_columns) != 507 or feature_columns != [f"Coef_{i}" for i in range(1, 508)]:
                        raise ValueError(f"Expected Coef_1..Coef_507, got {len(feature_columns)} in {feature_csv}")
                    required = {"Subject", "Group", "Class", "BinaryClass"}
                    if not required.issubset(source_columns):
                        raise ValueError(f"Missing labels in {feature_csv}: {sorted(required - set(source_columns))}")
                    if len(rows) != expected_counts.get((cohort, side, split), -1):
                        raise ValueError(
                            f"Row count differs from split manifest for {cohort}/{side}/{split}: "
                            f"{len(rows)} != {expected_counts.get((cohort, side, split))}"
                        )

                    subjects = []
                    for row in rows:
                        subject = (row.get("Subject") or "").strip()
                        if not subject:
                            raise ValueError(f"Empty Subject in {feature_csv}")
                        cls = int(row["Class"])
                        binary = int(row["BinaryClass"])
                        if cls not in (0, 1) or binary != int(cls == 1):
                            raise ValueError(f"Invalid class labels for {subject} in {feature_csv}")
                        for feature in feature_columns:
                            value = float(row[feature])
                            if value != value or value in (float("inf"), float("-inf")):
                                raise ValueError(f"Non-finite {feature} for {subject}")
                        subjects.append(subject)
                    if len(subjects) != len(set(subjects)):
                        raise ValueError(f"Duplicate subjects in {feature_csv}")

                    side_contract = train_contract if split == "test" else None
                    if split == "train":
                        train_contract = feature_contract
                        train_subjects = subjects
                    else:
                        test_contract = feature_contract
                        test_subjects = subjects
                    if side_contract is not None and side_contract != feature_contract:
                        raise ValueError(f"Train/test feature contract differs for {cohort}/{side}")

                    groups = patient_ids(subjects)
                    group_ids[(cohort, side, split)] = groups
                    output_dir = staging_root / cohort / side
                    output_dir.mkdir(parents=True, exist_ok=True)
                    target_csv = output_dir / f"{cohort}_{side.capitalize()}_{split}_coef_features.csv"
                    target_columns = META + feature_columns
                    with target_csv.open("w", newline="", encoding="utf-8") as stream:
                        writer = csv.DictWriter(stream, fieldnames=target_columns)
                        writer.writeheader()
                        for row in rows:
                            output_row = {key: row[key] for key in ("Subject", "Group", "Class", "BinaryClass")}
                            output_row["DataType"] = "Original"
                            output_row.update({key: row[key] for key in feature_columns})
                            writer.writerow(output_row)

                    output_payload = {
                        "csv_sha256": sha256(target_csv),
                        "feature_contract": feature_contract,
                        "dataset": cohort,
                        "side": side,
                        "split": split,
                        "protocol": "original-coef-no-pls-v1",
                        "rows": len(rows),
                        "schema": target_columns,
                        "source_feature_csv": str(feature_csv),
                        "source_feature_csv_sha256": sha256(feature_csv),
                        "pls_da_applied_to_model_input": False,
                        "synthetic_rows": 0,
                    }
                    Path(str(target_csv) + ".json").write_text(
                        json.dumps(output_payload, indent=2), encoding="utf-8"
                    )
                    contracts[(cohort, side, split)] = feature_contract
                    records.append({
                        "Dataset": cohort, "Side": side, "Split": split,
                        "Rows": len(rows), "Features": len(feature_columns),
                        "File": str(target_csv.relative_to(staging_root)),
                        "CSV_SHA256": sha256(target_csv),
                        "FeatureVariant": "ellalign",
                    })

                if group_ids[(cohort, side, "train")] & group_ids[(cohort, side, "test")]:
                    raise ValueError(f"Patient leakage in {cohort}/{side}")
                if train_contract != test_contract:
                    raise ValueError(f"Train/test feature contract differs for {cohort}/{side}")

        with (staging_root / "import_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        (staging_root / "IMPORT_MANIFEST.json").write_text(json.dumps({
            "schema": "spharm_split_model_inputs_v1",
            "created_at_local": datetime.now().astimezone().isoformat(),
            "source_split_root": str(split_root),
            "source_split_manifest_sha256": sha256(split_root / "split_manifest.json"),
            "output_root": str(output_root),
            "feature_variant": "ellalign",
            "model_input_protocol": "original-coef-no-pls-v1",
            "pls_da_applied_before_training": False,
            "training_note": "Use optuna_leakage_free_all.py with --protocol coef_plsda for fold-local PLS-DA augmentation, or coef_raw for the no-PLS baseline.",
            "test_policy": "Original-only; never augment or use for tuning/early stopping.",
            "records": records,
        }, indent=2), encoding="utf-8")
        staging_root.rename(output_root)
    except Exception:
        # Keep the staging directory for inspection; do not remove any files.
        raise

    print(f"Prepared {len(records)} training/test feature files under: {output_root}")
    print(f"Import manifest: {output_root / 'IMPORT_MANIFEST.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

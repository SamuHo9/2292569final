"""Build isolated Optuna input folders from a fixed-reference ICP rerun.

Expected inputs are the coefficient and XYZ CSVs produced separately from
left/right train/test SPHARM results. The builder filters the audited source
cohort column, verifies labels and provenance, and writes a fresh model-data
root without replacing the repository's historical inputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple


COHORTS = {
    "All_coef": None,
    "All_coef_Ds004469": "Ds004469",
    "All_coef_Ds005602": "Ds005602",
}
SIDES = ("left", "right")
SPLITS = ("train", "test")
RAW_PROTOCOL = "original-coef-no-pls-v1"
SUBJECT_RE = re.compile(r"(?i)(?:^|[_-])(sub-[a-z0-9]+)(?:[_-]|$)")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_bundle(path: Path) -> Tuple[List[str], List[dict], dict, str]:
    if not path.is_file():
        raise FileNotFoundError(f"Feature CSV not found: {path}")
    sidecar_path = Path(str(path) + ".json")
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"Feature sidecar not found: {sidecar_path}")
    payload = json.loads(sidecar_path.read_text(encoding="utf-8"))
    actual_hash = sha256(path)
    if payload.get("csv_sha256") != actual_hash:
        raise ValueError(f"CSV hash does not match sidecar: {path}")
    with path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        rows = list(reader)
        return list(reader.fieldnames), rows, payload, actual_hash


def patient_id(subject: str) -> str:
    match = SUBJECT_RE.search(str(subject))
    if not match:
        raise ValueError(f"Cannot find sub-* patient ID in feature Subject: {subject!r}")
    return match.group(1).lower()


def source_feature_paths(run_root: Path, side: str, split: str) -> Tuple[Path, Path]:
    icp_root = run_root / "icp" / side / f"{split}_fixed"
    feature_root = icp_root / "ml_features"
    return (feature_root / "spharm_results_coef_features.csv",
            feature_root / "spharm_xyz_coords.csv")


def feature_columns(header: Sequence[str], kind: str) -> List[str]:
    metadata = ({"Subject", "Group", "Class", "BinaryClass", "Dataset"}
                if kind == "coef" else
                {"Subject", "Group_Name", "Group_Label", "BinaryClass", "Dataset"})
    cols = [name for name in header if name not in metadata]
    if kind == "coef":
        expected = [f"Coef_{index}" for index in range(1, 508)]
    else:
        expected = [f"{axis}_{index}" for index in range(1002) for axis in "xyz"]
    if cols != expected:
        raise ValueError(f"Unexpected {kind} feature schema: expected {len(expected)} ordered columns, found {len(cols)}")
    return cols


def validate_source_contract(payload: Mapping[str, object], kind: str, path: Path) -> dict:
    contract = payload.get("feature_contract")
    if not isinstance(contract, dict):
        raise ValueError(f"Missing feature_contract in {path}.json")
    if kind == "coef":
        if contract.get("coefficient_variant") != "ellalign":
            raise ValueError(f"Expected ellalign coefficients in {path}")
        shared = contract
    else:
        if contract.get("mesh_variant") != "ellalign":
            raise ValueError(f"Expected ellalign point clouds in {path}")
        shared = contract.get("processing_provenance")
        if not isinstance(shared, dict):
            raise ValueError(f"Missing processing provenance in {path}.json")
    if shared.get("icp_mode") != "fixed_reference":
        raise ValueError(f"Expected fixed_reference ICP provenance in {path}")
    if not shared.get("icp_reference_sha256") or not shared.get("spharm_template_sha256"):
        raise ValueError(f"Missing ICP/SPHARM reference hash in {path}.json")
    return shared


def normalize_row(row: Mapping[str, str], kind: str, feature_names: Sequence[str]) -> dict:
    if kind == "coef":
        subject = str(row.get("Subject", "")).strip()
        group = str(row.get("Group", "")).strip()
        class_value = str(row.get("Class", "")).strip()
    else:
        subject = str(row.get("Subject", "")).strip()
        group = str(row.get("Group_Name", "")).strip()
        class_value = str(row.get("Group_Label", "")).strip()
    if not subject or not group or class_value not in {"0", "1", "2"}:
        raise ValueError(f"Invalid feature row metadata: {row}")
    cls = int(class_value)
    binary = int(str(row.get("BinaryClass", "")).strip())
    if binary not in (0, 1) or binary != int(cls == 1):
        raise ValueError(f"Class/BinaryClass inconsistency for {subject}")
    dataset = str(row.get("Dataset", "")).strip()
    if dataset not in {"Ds004469", "Ds005602"}:
        raise ValueError(f"Missing/invalid source Dataset for {subject}: {dataset!r}")
    features = {name: str(row[name]).strip() for name in feature_names}
    for name, value in features.items():
        try:
            if not math.isfinite(float(value)):
                raise ValueError
        except (TypeError, ValueError):
            raise ValueError(f"Invalid numeric feature {name} for {subject}: {value!r}") from None
    return {
        "Subject": subject, "Group": group, "Class": cls, "BinaryClass": binary,
        "Dataset": dataset, "features": features,
    }


def verify_pair(train: List[dict], test: List[dict], cohort: str, side: str) -> None:
    train_ids = [row["Subject"] for row in train]
    test_ids = [row["Subject"] for row in test]
    if len(set(train_ids)) != len(train_ids) or len(set(test_ids)) != len(test_ids):
        raise ValueError(f"Duplicate Subject rows in {cohort}/{side}")
    train_patients = {patient_id(value) for value in train_ids}
    test_patients = {patient_id(value) for value in test_ids}
    overlap = train_patients & test_patients
    if overlap:
        raise ValueError(f"Patient leakage in {cohort}/{side}: {sorted(overlap)[:10]}")
    for split, rows in (("train", train), ("test", test)):
        if not rows or {row["BinaryClass"] for row in rows} != {0, 1}:
            raise ValueError(f"Both classes are required in {cohort}/{side}/{split}")


def write_csv(path: Path, feature_names: Sequence[str], rows: Sequence[dict], kind: str) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["Subject", "Group", "Class", "BinaryClass"] + list(feature_names)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "Subject": row["Subject"], "Group": row["Group"],
                "Class": row["Class"], "BinaryClass": row["BinaryClass"],
                **row["features"],
            })
    return sha256(path)


def build(run_root: Path, output_root: Path) -> dict:
    run_root = run_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Model input output is not empty; choose a new folder: {output_root}")

    loaded = {}
    for side in SIDES:
        for split in SPLITS:
            coef_path, xyz_path = source_feature_paths(run_root, side, split)
            loaded[(side, split, "coef")] = read_bundle(coef_path)
            loaded[(side, split, "xyz")] = read_bundle(xyz_path)

    contracts = {}
    normalized = {}
    headers = {}
    source_hashes = {}
    for key, (header, rows, payload, csv_hash) in loaded.items():
        side, split, kind = key
        columns = feature_columns(header, kind)
        contract = validate_source_contract(payload, kind, source_feature_paths(run_root, side, split)[0 if kind == "coef" else 1])
        contracts[key] = contract
        headers[key] = columns
        source_hashes[key] = csv_hash
        normalized[key] = [normalize_row(row, kind, columns) for row in rows]

    for side in SIDES:
        for split in SPLITS:
            coef_rows = normalized[(side, split, "coef")]
            xyz_rows = normalized[(side, split, "xyz")]
            coef_map = {row["Subject"]: row for row in coef_rows}
            xyz_map = {row["Subject"]: row for row in xyz_rows}
            if set(coef_map) != set(xyz_map):
                raise ValueError(f"Coefficient/XYZ subject mismatch in {side}/{split}")
            for subject in coef_map:
                if (coef_map[subject]["BinaryClass"] != xyz_map[subject]["BinaryClass"] or
                    coef_map[subject]["Dataset"] != xyz_map[subject]["Dataset"]):
                    raise ValueError(f"Coefficient/XYZ label or cohort mismatch for {subject}")
        # Train/test must use the identical fixed ICP and SPHARM templates.
        for kind in ("coef", "xyz"):
            left_contract = contracts[(side, "train", kind)]
            right_contract = contracts[(side, "test", kind)]
            for key_name in ("icp_mode", "icp_reference_sha256", "spharm_template_sha256",
                             "spharm_degree", "subdiv_level", "grid_theta", "grid_phi"):
                if left_contract.get(key_name) != right_contract.get(key_name):
                    raise ValueError(f"Train/test {kind} processing contract differs for {side}: {key_name}")

    output_root.mkdir(parents=True, exist_ok=True)
    output_records = []
    for cohort, source_dataset in COHORTS.items():
        for side in SIDES:
            selected_by_split = {}
            for split in SPLITS:
                coef = [row for row in normalized[(side, split, "coef")]
                        if source_dataset is None or row["Dataset"] == source_dataset]
                xyz = [row for row in normalized[(side, split, "xyz")]
                       if source_dataset is None or row["Dataset"] == source_dataset]
                coef_map = {row["Subject"]: row for row in coef}
                xyz_map = {row["Subject"]: row for row in xyz}
                if set(coef_map) != set(xyz_map):
                    raise ValueError(f"Cohort-filtered coefficient/XYZ mismatch: {cohort}/{side}/{split}")
                selected_by_split[split] = coef
            verify_pair(selected_by_split["train"], selected_by_split["test"], cohort, side)

            for split in SPLITS:
                coef_rows = selected_by_split[split]
                xyz_rows = [row for row in normalized[(side, split, "xyz")]
                            if source_dataset is None or row["Dataset"] == source_dataset]
                coef_stem = f"{cohort}_{side.capitalize()}_{split}_coef_features.csv"
                coef_path = output_root / cohort / side / coef_stem
                coef_hash = write_csv(coef_path, headers[(side, split, "coef")], coef_rows, "coef")
                coef_contract = dict(contracts[(side, split, "coef")])
                coef_sidecar = {
                    "protocol": RAW_PROTOCOL, "cohort": cohort, "side": side, "split": split,
                    "source_feature_csv_sha256": source_hashes[(side, split, "coef")],
                    "csv_sha256": coef_hash, "feature_contract": coef_contract,
                    "pls_da_applied_to_model_input": False, "synthetic_rows": 0,
                    "label_definition": "BinaryClass=1 iff Class=1",
                }
                Path(str(coef_path) + ".json").write_text(
                    json.dumps(coef_sidecar, indent=2, ensure_ascii=False), encoding="utf-8")
                output_records.append({"kind": "coef", "cohort": cohort, "side": side,
                                      "split": split, "rows": len(coef_rows), "path": str(coef_path)})

                xyz_stem = {"All_coef": "ALL", "All_coef_Ds004469": "Ds004469",
                            "All_coef_Ds005602": "Ds005602"}[cohort]
                xyz_path = output_root / "Output_Dataset" / f"{xyz_stem}_{side.capitalize()}_{split}_xyz_coords.csv"
                xyz_hash = write_csv(xyz_path, headers[(side, split, "xyz")], xyz_rows, "xyz")
                xyz_contract = {
                    "version": "spharm-geometry-v1", "mesh_variant": "ellalign",
                    "num_points": 1002, "point_order": "x,y,z",
                    "processing_provenance": dict(contracts[(side, split, "xyz")]),
                }
                xyz_sidecar = {
                    "protocol": "original-xyz-no-pls-v1", "cohort": cohort, "side": side,
                    "split": split, "source_feature_csv_sha256": source_hashes[(side, split, "xyz")],
                    "csv_sha256": xyz_hash, "feature_contract": xyz_contract,
                    "synthetic_rows": 0, "label_definition": "BinaryClass=1 iff Class=1",
                }
                Path(str(xyz_path) + ".json").write_text(
                    json.dumps(xyz_sidecar, indent=2, ensure_ascii=False), encoding="utf-8")
                output_records.append({"kind": "xyz", "cohort": cohort, "side": side,
                                      "split": split, "rows": len(xyz_rows), "path": str(xyz_path)})

    manifest = {
        "schema": "fixed_reference_model_inputs_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_root": str(run_root), "output_root": str(output_root),
        "source_feature_csvs": {
            f"{side}/{split}/{kind}": str(source_feature_paths(run_root, side, split)[0 if kind == "coef" else 1])
            for side in SIDES for split in SPLITS for kind in ("coef", "xyz")
        },
        "patient_group_split_verified": True,
        "all_features_fixed_reference": True,
        "outputs": output_records,
    }
    manifest_path = output_root / "fixed_reference_model_inputs_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output_root": str(output_root), "files_written": len(output_records),
                      "manifest": str(manifest_path),
                      "row_counts": output_records}, indent=2, ensure_ascii=False))
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True,
                        help="Root produced by ICP/rerun_fixed_reference.py; SPHARM/features must exist below it")
    parser.add_argument("--output-root", type=Path, default=None,
                        help="Fresh model data root; default is <run-root>/model_inputs")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = args.output_root or (args.run_root / "model_inputs")
    build(args.run_root, output_root)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)

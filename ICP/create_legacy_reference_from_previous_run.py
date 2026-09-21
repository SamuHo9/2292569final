"""Recover fixed ICP references from the completed legacy groupwise runs.

The resulting references reproduce the old normalized coordinate frame. They
were fitted from all 381 participants, so they are for operational reruns and
reproducibility only, not independent held-out-test evaluation.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import shutil
import struct
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from reference_contract import LEGACY_GROUPWISE_VERSION, file_sha256, load_reference_contract


ICP_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = ICP_DIR / "references" / "legacy_groupwise_all_381_v1"
SIDES = ("left", "right")
EXPECTED_MODE = "exploratory_groupwise"
EXPECTED_SUBJECTS = 381
GEOMETRY_REL_TOL = 1e-5
SCALE_REL_TOL = 1e-5


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_nifti_grid(path: Path) -> tuple[tuple[int, int, int], tuple[float, float, float], int]:
    opener = gzip.open if path.name.lower().endswith(".gz") else open
    with opener(path, "rb") as stream:
        header = stream.read(348)
    if len(header) != 348:
        raise ValueError(f"NIfTI-1 header is truncated: {path}")
    little = struct.unpack("<i", header[:4])[0]
    big = struct.unpack(">i", header[:4])[0]
    if little == 348:
        endian = "<"
    elif big == 348:
        endian = ">"
    else:
        raise ValueError(f"Unrecognized NIfTI header size in {path}")
    dims = struct.unpack(endian + "8h", header[40:56])
    pixdim = struct.unpack(endian + "8f", header[76:108])
    grid = tuple(int(value) for value in dims[1:4])
    spacing = tuple(abs(float(value)) for value in pixdim[1:4])
    if any(value < 8 for value in grid) or any(not math.isfinite(value) or value <= 0 for value in spacing):
        raise ValueError(f"Invalid output grid in {path}: dim={grid}, spacing={spacing}")
    return grid, spacing, int(header[123])


def aligned_nifti_files(path: Path) -> list[Path]:
    files = [item for item in path.rglob("*") if item.is_file()
             and (item.name.lower().endswith(".nii.gz") or item.name.lower().endswith(".nii"))]
    return sorted(files, key=lambda item: item.name.lower())


def source_stem(name: str) -> str:
    lower = name.lower()
    suffix = ".nii.gz" if lower.endswith(".nii.gz") else ".nii"
    stem = name[:-len(suffix)]
    return stem[:-8] + suffix if stem.endswith("_aligned") else stem + suffix


def compare_with_current_status(side: str, legacy_names: list[str]) -> dict | None:
    status_path = ICP_DIR.parent / "SPHARM" / "split_data" / f"ALL_{side.title()}_file_status.csv"
    if not status_path.is_file():
        return None
    subject_pattern = re.compile(r"(?i)(?<![A-Za-z0-9])(?P<subject>sub-[A-Za-z0-9]+)(?![A-Za-z0-9])")
    legacy_ids = {match.group("subject").lower() for name in legacy_names
                  for match in subject_pattern.finditer(name)}
    current_ids = []
    with status_path.open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        if "FileName" not in (reader.fieldnames or []):
            raise ValueError(f"Missing FileName column in current status table: {status_path}")
        for row in reader:
            matches = list(subject_pattern.finditer(str(row.get("FileName", ""))))
            if len(matches) != 1:
                raise ValueError(f"Expected one subject ID in current status row: {row}")
            current_ids.append(matches[0].group("subject").lower())
    if len(set(current_ids)) != len(current_ids):
        raise ValueError(f"Duplicate subject IDs in current status table: {status_path}")
    overlap = set(current_ids) & legacy_ids
    return {
        "status_file": status_path.relative_to(ICP_DIR.parent).as_posix(),
        "status_file_sha256": file_sha256(status_path),
        "current_status_rows": len(current_ids),
        "current_unique_subjects": len(set(current_ids)),
        "legacy_reference_subjects": len(legacy_ids),
        "current_subjects_in_legacy_reference": len(overlap),
        "all_current_subjects_are_in_legacy_reference": set(current_ids).issubset(legacy_ids),
        "train_test_overlap_implication": (
            "Every subject in this current side status table was included in the old groupwise ICP reference."
            if set(current_ids).issubset(legacy_ids) else
            "At least one current status subject was not found in the old groupwise input list."
        ),
    }


def validate_side(side: str, source_root: Path, expected_subjects: int) -> dict:
    run_dir = source_root / f"output_{side}_hippocampus"
    status_path = run_dir / "icp_status.json"
    history_path = run_dir / "icp_convergence_history.json"
    matrix_path = run_dir / "T_matrices.npy"
    template_path = run_dir / "mean_shape.ply"
    nifti_dir = run_dir / "aligned_nifti"
    required = (status_path, history_path, matrix_path, template_path)
    missing = [str(path) for path in required if not path.is_file()]
    if missing or not nifti_dir.is_dir():
        raise FileNotFoundError(f"Legacy {side} ICP run is incomplete: {missing or [str(nifti_dir)]}")

    status = read_json(status_path)
    history = read_json(history_path)
    subjects = history.get("subject_names", [])
    matrices = np.load(str(matrix_path), allow_pickle=False)
    if status.get("success") is not True or status.get("mode") != EXPECTED_MODE:
        raise ValueError(f"Unexpected legacy ICP status for {side}: {status}")
    if int(status.get("subjects", -1)) != expected_subjects or len(subjects) != expected_subjects:
        raise ValueError(f"Expected {expected_subjects} subjects in {side} status/history")
    if len(set(subjects)) != len(subjects):
        raise ValueError(f"Duplicate subject names in {side} convergence history")
    if matrices.shape != (expected_subjects, 4, 4) or not np.isfinite(matrices).all():
        raise ValueError(f"Unexpected transform matrix shape/content for {side}: {matrices.shape}")

    singular_values = np.linalg.svd(matrices[:, :3, :3], compute_uv=False)
    per_subject_scale = singular_values.mean(axis=1)
    reference_scale = float(np.median(per_subject_scale))
    if reference_scale <= 0 or not np.isfinite(reference_scale):
        raise ValueError(f"Invalid recovered physical scale for {side}: {reference_scale}")
    max_relative_scale_error = float(np.max(np.abs(per_subject_scale - reference_scale)) / reference_scale)
    max_anisotropy = float(np.max(np.abs(singular_values - per_subject_scale[:, None])) / reference_scale)
    if max_relative_scale_error > SCALE_REL_TOL or max_anisotropy > SCALE_REL_TOL:
        raise ValueError(
            f"Transforms for {side} do not show one shared isotropic global scale: "
            f"scale_error={max_relative_scale_error:.3g}, anisotropy={max_anisotropy:.3g}"
        )

    outputs = aligned_nifti_files(nifti_dir)
    if len(outputs) != expected_subjects:
        raise ValueError(f"Expected {expected_subjects} aligned NIfTI files for {side}; found {len(outputs)}")
    observed_source_names = [source_stem(path.name) for path in outputs]
    if len(set(observed_source_names)) != expected_subjects or set(observed_source_names) != set(subjects):
        missing_names = sorted(set(subjects) - set(observed_source_names))
        extra_names = sorted(set(observed_source_names) - set(subjects))
        raise ValueError(
            f"Aligned NIfTI names do not match the recorded {side} input list "
            f"(missing={missing_names[:3]}, extra={extra_names[:3]})"
        )

    grids = [read_nifti_grid(path) for path in outputs]
    base_grid, base_spacing, base_units = grids[0]
    for path, (grid, spacing, units) in zip(outputs[1:], grids[1:]):
        if grid != base_grid or units != base_units or not np.allclose(
            spacing, base_spacing, rtol=GEOMETRY_REL_TOL, atol=1e-8
        ):
            raise ValueError(f"Inconsistent aligned NIfTI output geometry for {side}: {path.name}")
    if len(set(base_grid)) != 1 or not np.allclose(base_spacing, [base_spacing[0]] * 3,
                                                   rtol=GEOMETRY_REL_TOL, atol=1e-8):
        raise ValueError(f"Expected an isotropic cubic output grid for {side}; found {base_grid}/{base_spacing}")

    rounds = history.get("rounds", [])
    current_overlap = compare_with_current_status(side, subjects)
    return {
        "side": side,
        "run_dir": str(run_dir.resolve()),
        "status": status,
        "status_sha256": file_sha256(status_path),
        "history_sha256": file_sha256(history_path),
        "history_rounds": rounds,
        "subject_count": expected_subjects,
        "subject_list_sha256": sha256_bytes("\n".join(sorted(subjects)).encode("utf-8")),
        "current_dataset_overlap": current_overlap,
        "transform_sha256": file_sha256(matrix_path),
        "template_path": template_path,
        "template_sha256": file_sha256(template_path),
        "physical_to_normalized_scale": reference_scale,
        "max_relative_scale_error": max_relative_scale_error,
        "max_relative_anisotropy": max_anisotropy,
        "output_voxels": base_grid[0],
        "output_spacing": float(base_spacing[0]),
        "output_grid": list(base_grid),
        "output_spacing_xyz": list(base_spacing),
        "nifti_spatial_units_code": base_units,
        "aligned_nifti_count": len(outputs),
        "aligned_nifti_sample": outputs[0].name,
        "aligned_nifti_sample_sha256": file_sha256(outputs[0]),
    }


def create_references(source_root: Path, output_root: Path, expected_subjects: int) -> dict:
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Reference output must be new or empty: {output_root}")
    source_runs = [source_root / f"output_{side}_hippocampus" for side in SIDES]
    if any(run_dir == output_root or run_dir in output_root.parents for run_dir in source_runs):
        raise ValueError("Reference output cannot be placed inside the legacy source output folders")

    audited = {side: validate_side(side, source_root, expected_subjects) for side in SIDES}
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": "legacy_groupwise_icp_reference_bundle_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reference_version": LEGACY_GROUPWISE_VERSION,
        "source_method": "Completed exploratory groupwise ICP, then frozen mean_shape.ply",
        "source_scope": "All subjects in each old side-specific groupwise run",
        "independent_test_reference": False,
        "allowed_use": ["repeat the previous ICP coordinate convention", "operational rerun/inference compatibility"],
        "disallowed_use": ["claiming independent held-out test evaluation from this reference"],
        "parameter_note": (
            "Old icp_status.json did not preserve the full parameter set. Geometry/scale are recovered "
            "from stored transforms and all aligned NIfTI headers; new executions must use the explicit "
            "parameters in run_icp_with_reference.py and will record them in icp_status.json."
        ),
        "sides": {},
    }
    for side in SIDES:
        source = audited[side]
        side_dir = output_root / side
        side_dir.mkdir()
        template = side_dir / "mean_shape.ply"
        shutil.copy2(source["template_path"], template)
        contract = {
            "version": LEGACY_GROUPWISE_VERSION,
            "template_sha256": file_sha256(template),
            "physical_to_normalized_scale": source["physical_to_normalized_scale"],
            "output_spacing": source["output_spacing"],
            "output_voxels": source["output_voxels"],
            "reference_origin": "legacy_all_subject_groupwise_mean",
            "source_method": "exploratory_groupwise_v1_frozen_mean_shape",
            "source_subject_count": source["subject_count"],
            "source_subject_list_sha256": source["subject_list_sha256"],
            "source_status_sha256": source["status_sha256"],
            "source_history_sha256": source["history_sha256"],
            "source_transform_sha256": source["transform_sha256"],
            "source_template_sha256": source["template_sha256"],
            "source_aligned_nifti_sample_sha256": source["aligned_nifti_sample_sha256"],
            "source_grid": source["output_grid"],
            "source_spacing_xyz": source["output_spacing_xyz"],
            "physical_scale_derivation": "median(mean(svd(T_i[:3,:3]))) across all old subjects; validated as common isotropic scale",
            "independent_test_reference": False,
            "current_dataset_overlap": source["current_dataset_overlap"],
        }
        Path(str(template) + ".json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
        loaded = load_reference_contract(template)
        if loaded["version"] != LEGACY_GROUPWISE_VERSION:
            raise RuntimeError(f"Reference contract version did not round-trip for {side}")
        manifest["sides"][side] = {
            key: value for key, value in source.items() if key not in {"template_path", "status"}
        }
        manifest["sides"][side].update({
            "reference_template": str(template.relative_to(output_root)),
            "reference_metadata": str(Path(str(template) + ".json").relative_to(output_root)),
            "reference_sha256": loaded["template_sha256"],
        })
    manifest_path = output_root / "reference_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ICP_DIR,
                        help="Directory containing output_left_hippocampus and output_right_hippocampus")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--expected-subjects", type=int, default=EXPECTED_SUBJECTS)
    args = parser.parse_args()
    manifest = create_references(args.source_root, args.output_root, args.expected_subjects)
    print(json.dumps({
        "reference_root": str(args.output_root.resolve()),
        "independent_test_reference": manifest["independent_test_reference"],
        "sides": {
            side: {
                "subjects": manifest["sides"][side]["subject_count"],
                "reference_sha256": manifest["sides"][side]["reference_sha256"],
                "scale": manifest["sides"][side]["physical_to_normalized_scale"],
                "grid": manifest["sides"][side]["output_grid"],
                "spacing": manifest["sides"][side]["output_spacing"],
            } for side in SIDES
        },
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

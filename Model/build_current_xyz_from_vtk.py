"""Build the current 80/20 raw-XYZ PointNet inputs from SPHARM VTK meshes.

The coefficient split is the source of truth for subject membership, labels and
train/test assignment.  This script only converts the matching ellalign VTK
points to CSV; it does not augment, fit PLS-DA, or read any test data while
training.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import vtk


DATASET_MAP = {
    "All_coef": "ALL",
    "All_coef_Ds004469": "Ds004469",
    "All_coef_Ds005602": "Ds005602",
}
SIDES = ("left", "right")
SPLITS = ("train", "test")
META = ("Subject", "Group", "Class", "BinaryClass", "DataType")
POINTS = 1002


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        required = set(META)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing metadata columns {sorted(missing)}: {path}")
        return list(reader)


def vtk_points(path: Path):
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    poly = reader.GetOutput()
    if poly is None or poly.GetPoints() is None:
        raise ValueError(f"No points in VTK: {path}")
    if poly.GetNumberOfPoints() != POINTS:
        raise ValueError(
            f"Expected {POINTS} points, got {poly.GetNumberOfPoints()}: {path}"
        )
    return [poly.GetPoint(index) for index in range(POINTS)]


def write_split(dataset, source_dataset, side, split, rows, vtk_root, output_root):
    stem = f"{source_dataset}_{side.capitalize()}"
    output_path = output_root / f"{stem}_{split}_xyz_coords.csv"
    records = []
    feature_names = [
        f"{axis}_{index}"
        for index in range(POINTS)
        for axis in ("x", "y", "z")
    ]
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow([*META, *feature_names])
        for row in rows:
            subject = str(row["Subject"])
            vtk_path = vtk_root / side / "spharm_results" / (
                subject + "_SPHARM_ellalign.vtk"
            )
            if not vtk_path.is_file():
                raise FileNotFoundError(f"Missing matching VTK: {vtk_path}")
            points = vtk_points(vtk_path)
            values = []
            for point in points:
                values.extend(f"{float(value):.8f}" for value in point)
            writer.writerow([*(row[name] for name in META), *values])
            records.append({
                "Subject": subject,
                "vtk": str(vtk_path),
                "vtk_sha256": sha256_file(vtk_path),
            })
    sidecar = {
        "protocol": "original-xyz-no-pls-v1",
        "cohort": source_dataset,
        "source_cohort": dataset,
        "side": side,
        "split": split,
        "feature_kind": "xyz",
        "mesh_variant": "ellalign",
        "num_points": POINTS,
        "point_order": "x,y,z",
        "pls_da_applied_to_model_input": False,
        "synthetic_rows": 0,
        "rows": len(rows),
        "feature_contract": {
            "mesh_variant": "ellalign",
            "coefficient_variant": "ellalign",
            "num_points": POINTS,
            "point_order": "x,y,z",
        },
        "source_vtk_root": str(vtk_root.resolve()),
        "source_coefficient_csv": str(
            (Path(__file__).resolve().parent / "current_spharm_80_20_20260922" /
             dataset / side / f"{dataset}_{side.capitalize()}_{split}_coef_features.csv").resolve()
        ),
        "subjects": records,
    }
    Path(str(output_path) + ".json").write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return output_path, len(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        default=str(Path(__file__).resolve().parent / "current_spharm_80_20_20260922"),
    )
    parser.add_argument(
        "--vtk_root",
        default=str(
            Path(__file__).resolve().parent.parent
            / "reruns" / "icp_legacy_reference_20260921_172727"
        ),
    )
    parser.add_argument("--output_root", default=None)
    args = parser.parse_args()
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else data_root / "Output_Dataset"
    output_root.mkdir(parents=True, exist_ok=True)
    built = []
    for dataset, source_dataset in DATASET_MAP.items():
        for side in SIDES:
            for split in SPLITS:
                coefficient_path = (
                    data_root / dataset / side /
                    f"{dataset}_{side.capitalize()}_{split}_coef_features.csv"
                )
                rows = read_rows(coefficient_path)
                output_path, count = write_split(
                    dataset, source_dataset, side, split, rows,
                    Path(args.vtk_root).resolve(), output_root
                )
                built.append({
                    "dataset": dataset,
                    "source_dataset": source_dataset,
                    "side": side,
                    "split": split,
                    "rows": count,
                    "path": str(output_path),
                })
                print(f"Wrote {output_path} ({count} rows)", flush=True)
    manifest = {
        "schema": "current_spharm_xyz_pointnet_inputs_v1",
        "protocol": "original-xyz-no-pls-v1",
        "source_coefficient_root": str(data_root),
        "source_vtk_root": str(Path(args.vtk_root).resolve()),
        "mesh_variant": "ellalign",
        "pls_da_applied_before_training": False,
        "records": built,
    }
    (output_root / "XYZ_IMPORT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Wrote {output_root / 'XYZ_IMPORT_MANIFEST.json'}", flush=True)


if __name__ == "__main__":
    main()

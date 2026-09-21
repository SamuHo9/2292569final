#!/usr/bin/env python3
"""Run one fixed-reference ICP step through the normal Python/Slicer bootstrap.

From the project root, the default inputs are:
  ../merged_ds005602_ds004469/left_hippocampus
  ../merged_ds005602_ds004469/right_hippocampus

Examples:
  python ICP/run_icp_with_reference.py --side left --check-only
  python ICP/run_icp_with_reference.py --side left
  python ICP/run_icp_with_reference.py --side right
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ICP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ICP_DIR.parent
ICP_SCRIPT = ICP_DIR / "ICP.py"
DEFAULT_INPUT_ROOT = PROJECT_ROOT.parent / "merged_ds005602_ds004469"
DEFAULT_REFERENCE_ROOT = ICP_DIR / "references" / "legacy_groupwise_all_381_v1"
DEFAULT_SLICER_EXE = Path(r"C:\Program Files\SlicerSALT 6.0.0\SlicerSALT.exe")
LEGACY_REFERENCE_VERSION = "fixed-legacy-groupwise-reference-v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def supported_volume(path):
    name = path.name.lower()
    return name.endswith((".nii.gz", ".nii", ".hdr", ".nrrd"))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Align one hemisphere to its audited legacy ICP reference using SlicerSALT."
    )
    parser.add_argument("--side", required=True, choices=("left", "right"))
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input-dir",
        type=Path,
        help="Batch input folder. Defaults to the selected side under the sibling merged dataset.",
    )
    input_group.add_argument(
        "--input-file",
        type=Path,
        help="Run one mask file instead of a batch.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        help="Parent of left_hippocampus/right_hippocampus; defaults to the sibling merged dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Fresh output folder. If omitted, a timestamped folder is created under reruns/.",
    )
    parser.add_argument(
        "--run-id",
        help="Optional shared run folder name, so separate left/right commands can share one rerun root.",
    )
    parser.add_argument(
        "--reference-root",
        type=Path,
        default=DEFAULT_REFERENCE_ROOT,
        help="Reference bundle root (defaults to ICP/references/legacy_groupwise_all_381_v1).",
    )
    parser.add_argument(
        "--slicer-exe",
        type=Path,
        default=DEFAULT_SLICER_EXE,
        help="SlicerSALT executable path.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate paths, side, and reference without starting ICP.",
    )
    return parser.parse_args(argv)


def resolve_inputs(args):
    if args.input_file is not None:
        input_file = args.input_file.expanduser().resolve()
        if not input_file.is_file():
            raise FileNotFoundError("Input file not found: {}".format(input_file))
        if not supported_volume(input_file):
            raise ValueError("Unsupported input volume type: {}".format(input_file.name))
        input_files = [input_file]
        input_path = input_file
        input_mode = "single"
    else:
        if args.input_dir is not None:
            input_dir = args.input_dir.expanduser().resolve()
        else:
            input_root = (args.input_root or DEFAULT_INPUT_ROOT).expanduser().resolve()
            input_dir = input_root / (args.side + "_hippocampus")
        if not input_dir.is_dir():
            raise FileNotFoundError("Input directory not found: {}".format(input_dir))
        input_files = sorted(
            (path for path in input_dir.rglob("*") if path.is_file() and supported_volume(path)),
            key=lambda path: path.name.casefold(),
        )
        labelled = [path for path in input_files if "label" in path.name.casefold()]
        if labelled:
            input_files = labelled
        if not input_files:
            raise ValueError("No supported NIfTI/NRRD input volumes found in {}".format(input_dir))
        input_path = input_dir
        input_mode = "batch"

    opposite_markers = "right|rh" if args.side == "left" else "left|lh"
    opposite_pattern = re.compile(
        r"(?i)(?:^|[_-])(?:{})(?:[_-]|\.)".format(opposite_markers)
    )
    wrong_side = [path.name for path in input_files if opposite_pattern.search(path.name)]
    if wrong_side:
        raise ValueError(
            "Input side does not match --side {}. Opposite-side file: {}".format(
                args.side, wrong_side[0]
            )
        )
    return input_mode, input_path, input_files


def load_reference(args):
    reference_root = args.reference_root.expanduser().resolve()
    reference_template = reference_root / args.side / "mean_shape.ply"
    reference_metadata = Path(str(reference_template) + ".json")
    if not reference_template.is_file() or not reference_metadata.is_file():
        raise FileNotFoundError(
            "Reference bundle is incomplete for {}: {} and {}".format(
                args.side, reference_template, reference_metadata
            )
        )

    contract = json.loads(reference_metadata.read_text(encoding="utf-8"))
    if contract.get("version") != LEGACY_REFERENCE_VERSION:
        raise ValueError("Unexpected reference version: {}".format(contract.get("version")))
    if contract.get("independent_test_reference") is not False:
        raise ValueError("Legacy reference provenance must mark independent_test_reference=false")
    actual_hash = sha256_file(reference_template)
    if contract.get("template_sha256") != actual_hash:
        raise ValueError("Reference SHA-256 does not match its metadata: {}".format(reference_template))
    if not contract.get("output_voxels") or not contract.get("output_spacing"):
        raise ValueError("Reference metadata is missing the output grid definition")
    return reference_template, contract, actual_hash


def make_output_dir(args, input_mode, input_path):
    if args.output_dir is not None:
        return args.output_dir.expanduser().resolve()
    run_id = args.run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_id):
        raise ValueError("--run-id may contain only letters, numbers, dot, underscore, and hyphen")
    run_root = PROJECT_ROOT / "reruns" / ("icp_legacy_reference_" + run_id)
    if input_mode == "single":
        case_name = input_path.name
        if case_name.lower().endswith(".nii.gz"):
            case_name = case_name[:-7]
        else:
            case_name = input_path.stem
        subject_match = re.search(r"sub-[A-Za-z0-9-]+", case_name, flags=re.IGNORECASE)
        if subject_match:
            case_name = subject_match.group(0)
        else:
            case_name = re.sub(r"^(left|right)[_-]", "", case_name, flags=re.IGNORECASE)
        return run_root / ("single_{}_{}".format(args.side, case_name))
    return run_root / args.side


def build_icp_arguments(args, input_mode, input_path, output_dir, reference_template, contract):
    arguments = [
        "--output_dir", str(output_dir),
        "--reference_template", str(reference_template),
        "--output_voxels", str(int(contract["output_voxels"])),
        "--output_spacing", str(float(contract["output_spacing"])),
        "--max_iterations", "20",
        "--tolerance", "0.00005",
        "--pairwise_iterations", "100",
        "--pairwise_tolerance", "0.0001",
        "--pairwise_landmarks", "200",
        "--interpolation", "nn",
        "--invert_transform", "auto",
    ]
    if input_mode == "single":
        arguments.extend(("--input_file", str(input_path)))
    else:
        arguments.extend(("--input_dir", str(input_path)))
    return arguments


def verify_outputs(args, input_mode, input_path, expected_count, output_dir, reference_template,
                   contract, reference_hash, process_returncode):
    status_path = output_dir / "icp_status.json"
    log_path = output_dir / "icp_debug_log.txt"
    if not status_path.is_file():
        raise RuntimeError(
            "ICP.py/Slicer finished without icp_status.json (exit code {}). Check {}".format(
                process_returncode, log_path
            )
        )

    status = json.loads(status_path.read_text(encoding="utf-8"))
    if status.get("success") is not True or status.get("mode") != "fixed_reference":
        raise RuntimeError("ICP status is not a successful fixed-reference run: {}".format(status))
    if status.get("reference_sha256") != reference_hash:
        raise RuntimeError("ICP status reports a different reference hash")
    if status.get("geometry_version") != LEGACY_REFERENCE_VERSION:
        raise RuntimeError("ICP status reports an unexpected geometry version")
    if status.get("independent_test_reference") is not False:
        raise RuntimeError("ICP status lost the legacy-reference provenance flag")
    if int(status.get("subjects", -1)) != expected_count:
        raise RuntimeError(
            "ICP status count mismatch: expected {}, got {}".format(
                expected_count, status.get("subjects")
            )
        )
    aligned_dir = output_dir / "aligned_nifti"
    aligned_count = sum(
        1 for path in aligned_dir.iterdir()
        if path.is_file() and path.name.lower().endswith((".nii", ".nii.gz"))
    ) if aligned_dir.is_dir() else 0
    if aligned_count != expected_count:
        raise RuntimeError(
            "Aligned output count mismatch: expected {}, found {} in {}".format(
                expected_count, aligned_count, aligned_dir
            )
        )
    if process_returncode != 0:
        raise RuntimeError(
            "ICP outputs passed count checks, but Python/Slicer returned exit code {}".format(
                process_returncode
            )
        )

    summary = {
        "schema": "fixed_reference_icp_run_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "side": args.side,
        "input_mode": input_mode,
        "input_path": str(input_path),
        "subjects": aligned_count,
        "reference": str(reference_template),
        "reference_sha256": reference_hash,
        "reference_version": contract["version"],
        "independent_test_reference": False,
        "output": str(output_dir),
        "parameters": status.get("parameters", {}),
    }
    (output_dir / "run_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return aligned_count


def main(argv=None):
    args = parse_args(argv)
    try:
        if args.input_root is not None and args.input_dir is not None:
            raise ValueError("Use either --input-root or --input-dir, not both")
        input_mode, input_path, input_files = resolve_inputs(args)
        reference_template, contract, reference_hash = load_reference(args)
        slicer_exe = args.slicer_exe.expanduser().resolve()
        if not slicer_exe.is_file():
            raise FileNotFoundError("SlicerSALT executable not found: {}".format(slicer_exe))
        output_dir = make_output_dir(args, input_mode, input_path)
        if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
            raise FileExistsError(
                "Output directory must be new or empty: {}".format(output_dir)
            )

        print("ICP step: fixed legacy reference ({}, {})".format(args.side, input_mode), flush=True)
        print("Input: {}".format(input_path), flush=True)
        print("Input volumes: {}".format(len(input_files)), flush=True)
        print("Reference: {}".format(reference_template), flush=True)
        print("Reference SHA-256: {}".format(reference_hash), flush=True)
        print("Output: {}".format(output_dir), flush=True)
        print(
            "Note: this reference was built from the prior all-subject groupwise run; "
            "it is not independent of the held-out test subjects.",
            flush=True,
        )
        if args.check_only:
            print("Check passed. ICP was not started.", flush=True)
            return 0

        output_dir.mkdir(parents=True, exist_ok=True)
        icp_arguments = build_icp_arguments(
            args, input_mode, input_path, output_dir, reference_template, contract
        )
        command = [
            str(slicer_exe),
            "--no-main-window",
            "--no-splash",
            "--python-script",
            str(ICP_SCRIPT),
        ] + icp_arguments
        environment = os.environ.copy()
        print("Starting SlicerSALT with ICP.py; waiting for Slicer to finish...", flush=True)
        completed = subprocess.run(command, cwd=str(PROJECT_ROOT), env=environment, check=False)
        count = verify_outputs(
            args, input_mode, input_path, len(input_files), output_dir,
            reference_template, contract, reference_hash, completed.returncode
        )
        print("Verified {} aligned volume(s).".format(count), flush=True)
        print("Run summary: {}".format(output_dir / "run_summary.json"), flush=True)
        return 0
    except Exception as error:
        print("ERROR: {}".format(error), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Stage audited train/test masks, fit ICP references from train only, and align both splits.

Run from Git Bash through SlicerSALT's PythonSlicer executable. This is the Python
counterpart of the former PowerShell orchestration script.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys


ICP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ICP_DIR.parent
ICP_SCRIPT = ICP_DIR / "ICP.py"
PREPARE_SCRIPT = ICP_DIR / "prepare_mask_split.py"
DEFAULT_SPLIT_MANIFEST = PROJECT_ROOT / "SPHARM" / "split_data" / "current_split_manifest.json"
DEFAULT_LEFT_STATUS = PROJECT_ROOT / "SPHARM" / "split_data" / "ALL_Left_file_status.csv"
DEFAULT_RIGHT_STATUS = PROJECT_ROOT / "SPHARM" / "split_data" / "ALL_Right_file_status.csv"
DEFAULT_SLICER_EXE = Path(r"C:\Program Files\SlicerSALT 6.0.0\SlicerSALT.exe")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path, description):
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError("{} not found: {}".format(description, path))
    return path


def require_mask_dir(path, side):
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError("{} mask folder not found: {}".format(side, path))
    masks = sorted(path.glob("*.nii.gz"))
    if not masks:
        raise ValueError("No .nii.gz masks found for {} in {}".format(side, path))
    return path, len(masks)


def check_output_root(path):
    path = Path(path).expanduser().resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(
            "Output root must be new or empty; choose a fresh run folder: {}".format(path)
        )
    return path


def read_status(job_output, expected_mode):
    status_path = job_output / "icp_status.json"
    if not status_path.is_file():
        raise RuntimeError("ICP did not write status file: {}".format(status_path))
    status = json.loads(status_path.read_text(encoding="utf-8-sig"))
    if status.get("success") is not True or status.get("mode") != expected_mode:
        raise RuntimeError(
            "ICP status is not successful for {}: {}".format(job_output, status)
        )
    aligned_dir = job_output / "aligned_nifti"
    count = sum(
        1 for item in aligned_dir.glob("*.nii.gz") if item.is_file()
    ) if aligned_dir.is_dir() else 0
    expected_count = int(status.get("subjects", -1))
    if count <= 0 or count != expected_count:
        raise RuntimeError(
            "Aligned NIfTI count mismatch in {}: status={}, files={}".format(
                aligned_dir, expected_count, count
            )
        )
    return status, count


def invoke_icp(slicer_exe, input_dir, output_dir, mode, reference_template=None):
    arguments = [
        str(slicer_exe),
        "--no-main-window", "--no-splash", "--python-script", str(ICP_SCRIPT),
        "--input_dir", str(input_dir), "--output_dir", str(output_dir),
        "--output_voxels", "128", "--max_iterations", "20", "--tolerance", "0.00005",
        "--pairwise_iterations", "100", "--pairwise_tolerance", "0.0001",
        "--pairwise_landmarks", "200", "--interpolation", "nn",
    ]
    if mode == "fit_reference":
        arguments.append("--fit_reference")
    elif mode == "fixed_reference":
        if reference_template is None:
            raise ValueError("fixed_reference requires a reference template")
        arguments.extend(("--reference_template", str(reference_template)))
    else:
        raise ValueError("Unsupported ICP mode: {}".format(mode))

    print("\nICP {}: {}".format(mode, input_dir), flush=True)
    print("Output: {}".format(output_dir), flush=True)
    process = subprocess.run(arguments, cwd=str(PROJECT_ROOT), check=False)
    status, count = read_status(output_dir, mode)
    if process.returncode != 0:
        print(
            "WARNING: Slicer returned exit code {}, but status and output-count checks passed: {}".format(
                process.returncode, output_dir
            ),
            file=sys.stderr,
            flush=True,
        )
    return status, count


def prepare_masks(args, python_exe, output_root):
    command = [
        str(python_exe), str(PREPARE_SCRIPT),
        "--left-source", str(args.left_mask_dir),
        "--right-source", str(args.right_mask_dir),
        "--split-manifest", str(args.split_manifest),
        "--left-status", str(args.left_status),
        "--right-status", str(args.right_status),
        "--output-root", str(output_root),
    ]
    print("\n[1/4] Copying audited patient-level train/test masks...", flush=True)
    subprocess.run(command, cwd=str(PROJECT_ROOT), check=True)
    staged_manifest = output_root / "pre_icp_split_manifest.json"
    if not staged_manifest.is_file():
        raise RuntimeError("Mask staging did not write {}".format(staged_manifest))
    return staged_manifest


def run(args):
    require_file(ICP_SCRIPT, "ICP script")
    require_file(PREPARE_SCRIPT, "Mask split preparation script")
    left_dir, left_count = require_mask_dir(args.left_mask_dir, "left")
    right_dir, right_count = require_mask_dir(args.right_mask_dir, "right")
    split_manifest = require_file(args.split_manifest, "Split manifest")
    left_status = require_file(args.left_status, "Left status table")
    right_status = require_file(args.right_status, "Right status table")
    slicer_exe = require_file(args.slicer_exe, "SlicerSALT executable")
    python_exe = require_file(args.python_exe, "Python executable")
    output_root = check_output_root(args.output_root)

    print("Project: {}".format(PROJECT_ROOT))
    print("Input masks: left={} ({} files), right={} ({} files)".format(
        left_dir, left_count, right_dir, right_count
    ))
    print("Split manifest: {}".format(split_manifest))
    print("Output root: {}".format(output_root))
    if args.check_only:
        print("Preflight passed. No masks were copied and no ICP job was started.")
        return 0

    args.left_mask_dir = left_dir
    args.right_mask_dir = right_dir
    args.split_manifest = split_manifest
    args.left_status = left_status
    args.right_status = right_status
    staged_manifest = prepare_masks(args, python_exe, output_root)
    jobs = []

    for side in ("left", "right"):
        fit_input = output_root / "masks" / side / "train"
        fit_output = output_root / "icp" / side / "reference_fit"
        print("\n[2/4] Fitting {} reference from training masks only...".format(side), flush=True)
        fit_status, fit_count = invoke_icp(
            slicer_exe, fit_input, fit_output, "fit_reference"
        )
        template = fit_output / "mean_shape.ply"
        contract_path = Path(str(template) + ".json")
        if not template.is_file() or not contract_path.is_file():
            raise RuntimeError("Training reference or metadata is missing: {}".format(template))
        template_hash = sha256_file(template)
        contract = json.loads(contract_path.read_text(encoding="utf-8-sig"))
        if contract.get("template_sha256") != template_hash:
            raise RuntimeError("Reference metadata hash mismatch for {}".format(side))
        if fit_status.get("reference_sha256") != template_hash:
            raise RuntimeError("ICP fit status hash mismatch for {}".format(side))
        jobs.append({
            "side": side,
            "split": "train_reference_fit",
            "mode": fit_status["mode"],
            "subjects": fit_count,
            "reference": str(template),
            "reference_sha256": template_hash,
            "output": str(fit_output),
        })

        print("\n[3/4] Aligning {} train/test masks to its frozen train reference...".format(side), flush=True)
        for split in ("train", "test"):
            input_dir = output_root / "masks" / side / split
            job_output = output_root / "icp" / side / (split + "_fixed")
            status, count = invoke_icp(
                slicer_exe, input_dir, job_output, "fixed_reference", template
            )
            if status.get("reference_sha256") != template_hash:
                raise RuntimeError(
                    "{} {} did not use the frozen training reference".format(side, split)
                )
            jobs.append({
                "side": side,
                "split": split,
                "mode": status["mode"],
                "subjects": count,
                "reference": str(template),
                "reference_sha256": status["reference_sha256"],
                "output": str(job_output),
            })

    report = {
        "schema": "fixed_reference_icp_rerun_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_manifest": str(split_manifest),
        "staged_manifest": str(staged_manifest),
        "jobs": jobs,
        "next_step": (
            "Run SPHARM separately for each side/split using its fixed production template; "
            "use each ICP job directory as the SPHARM output root."
        ),
    }
    report_path = output_root / "fixed_reference_icp_rerun_manifest.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("\n[4/4] Fixed-reference ICP finished and verified.", flush=True)
    print("Run report: {}".format(report_path), flush=True)
    return 0


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-mask-dir", type=Path, required=True)
    parser.add_argument("--right-mask-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--left-status", type=Path, default=DEFAULT_LEFT_STATUS)
    parser.add_argument("--right-status", type=Path, default=DEFAULT_RIGHT_STATUS)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python-exe", type=Path, default=Path(sys.executable))
    parser.add_argument("--slicer-exe", type=Path, default=DEFAULT_SLICER_EXE)
    parser.add_argument("--check-only", action="store_true")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, FileExistsError, ValueError, OSError,
            subprocess.CalledProcessError, json.JSONDecodeError, RuntimeError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        raise SystemExit(1)

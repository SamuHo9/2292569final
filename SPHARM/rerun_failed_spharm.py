#!/usr/bin/env python3
"""Retry incomplete SPHARM subjects with their original side-specific template.

The retry writes into a new, isolated directory first. Only subjects that pass
the deep verifier are copied into the original spharm_results directory.
Original shard status files and logs are preserved as the first-attempt record.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

from verify_spharm_outputs import verify


DEFAULT_SLICER = Path(r"C:\Program Files\SlicerSALT 6.0.0\SlicerSALT.exe")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
REQUIRED_SUFFIXES = (
    "_SPHARM.coef",
    "_SPHARM.vtk",
    "_SPHARM_grid.vtk",
    "_SPHARM_ellalign.coef",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def subject_stem(filename: str) -> str:
    return filename[:-7] if filename.lower().endswith(".nii.gz") else Path(filename).stem


def load_side_plan(run_root: Path, side: str) -> dict:
    side_root = run_root / side
    status_files = sorted(side_root.glob("spharm_status_shard*.json"))
    if not status_files:
        raise RuntimeError(f"No SPHARM shard status files found for {side}: {side_root}")

    statuses = [read_json(path) for path in status_files]
    templates = {status.get("reference_template") for status in statuses}
    template_hashes = {status.get("reference_sha256") for status in statuses}
    configs = {json.dumps(status.get("config", {}), sort_keys=True) for status in statuses}
    if len(templates) != 1 or len(template_hashes) != 1 or len(configs) != 1:
        raise RuntimeError(f"Mixed SPHARM template/config found in {side} shard statuses")

    template_text = next(iter(templates))
    expected_hash = next(iter(template_hashes))
    template = Path(template_text)
    if not template.is_file():
        raise FileNotFoundError(f"Original {side} reference template not found: {template}")
    actual_hash = sha256(template)
    if actual_hash.lower() != str(expected_hash).lower():
        raise RuntimeError(
            f"Original {side} template hash changed: expected {expected_hash}, got {actual_hash}"
        )

    status_failed = {
        Path(item["input"]).name
        for status in statuses
        for item in status.get("subjects", [])
        if not item.get("success", False)
    }
    report_path = side_root / "verify_spharm_outputs.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"Run the SPHARM verifier first: {report_path}")
    report = read_json(report_path)
    report_failed = {
        f"{item['subject']}.nii.gz"
        for item in report.get("missing", []) + report.get("invalid_geometry", [])
    }
    exclusion_path = run_root / "spharm_exclusion_manifest.json"
    excluded_names: set[str] = set()
    if exclusion_path.is_file():
        exclusion = read_json(exclusion_path)
        if exclusion.get("schema") != "spharm_case_exclusion_v1":
            raise RuntimeError(f"Unsupported SPHARM exclusion manifest: {exclusion_path}")
        excluded_names = {
            Path(name).name
            for name in exclusion.get("excluded_by_side", {}).get(side, [])
        }

    failed_names = sorted((status_failed | report_failed) - excluded_names)
    if not failed_names:
        return {
            "side": side,
            "side_root": side_root,
            "input_dir": side_root / "aligned_nifti",
            "template": template,
            "template_sha256": actual_hash,
            "failed_names": [],
            "excluded_names": sorted(excluded_names),
            "config": json.loads(next(iter(configs))),
        }

    input_dir = side_root / "aligned_nifti"
    missing_inputs = [name for name in failed_names if not (input_dir / name).is_file()]
    if missing_inputs:
        raise FileNotFoundError(f"Missing {side} failed inputs: {missing_inputs}")

    return {
        "side": side,
        "side_root": side_root,
        "input_dir": input_dir,
        "template": template,
        "template_sha256": actual_hash,
        "failed_names": failed_names,
        "excluded_names": sorted(excluded_names),
        "config": json.loads(next(iter(configs))),
    }


def inspect_retry(input_dir: Path, output_dir: Path, report_path: Path) -> dict:
    result = verify(input_dir, output_dir, deep=True)
    write_json(report_path, result)
    return result


def run_side(plan: dict, retry_root: Path, slicer_exe: Path, workers: int) -> dict:
    side = plan["side"]
    if not plan["failed_names"]:
        return {"side": side, "attempted": 0, "runner_exit_code": 0, "retry_report": None,
                "merged_subjects": [], "merge_conflicts": []}

    retry_side = retry_root / side
    retry_input = retry_side / "input"
    retry_output = retry_side / "output"
    retry_input.mkdir(parents=True, exist_ok=False)
    retry_output.mkdir(parents=True, exist_ok=False)

    for filename in plan["failed_names"]:
        shutil.copy2(plan["input_dir"] / filename, retry_input / filename)

    # Preserve the original ICP provenance for SPHARM processing sidecars.
    icp_status = plan["side_root"] / "icp_status.json"
    if icp_status.is_file():
        shutil.copy2(icp_status, retry_output / "icp_status.json")

    runner = SCRIPT_DIR / "run_spharm_parallel.py"
    command = [
        sys.executable,
        str(runner),
        "--slicer_exe",
        str(slicer_exe),
        "--num_workers",
        str(workers),
        "--input_dir",
        str(retry_input),
        "--output_dir",
        str(retry_output),
        "--reference_template",
        str(plan["template"]),
    ]
    print(f"\n=== Retry SPHARM {side}: {len(plan['failed_names'])} failed input(s), {workers} worker(s) ===", flush=True)
    print(f"Reference: {plan['template']}", flush=True)
    print(f"Reference SHA-256: {plan['template_sha256']}", flush=True)
    print(f"Retry output: {retry_output}", flush=True)
    runner_result = subprocess.run(command, check=False)

    retry_report_path = retry_output / "verify_spharm_outputs.json"
    retry_report = inspect_retry(retry_input, retry_output, retry_report_path)
    retry_failed_stems = {
        item["subject"] for item in retry_report.get("missing", [])
    } | {
        item["subject"] for item in retry_report.get("invalid_geometry", [])
    }
    successful_stems = {subject_stem(name) for name in plan["failed_names"]} - retry_failed_stems

    source_results = retry_output / "spharm_results"
    target_results = plan["side_root"] / "spharm_results"
    target_results.mkdir(parents=True, exist_ok=True)
    merged_subjects = []
    merge_conflicts = []
    for stem in sorted(successful_stems):
        conflicts = []
        for source in source_results.glob(f"{stem}_*"):
            if not source.is_file() or source.name.endswith("_pp.nrrd"):
                continue
            target = target_results / source.name
            if target.exists():
                if target.stat().st_size == source.stat().st_size and sha256(target) == sha256(source):
                    continue
                conflicts.append(source.name)
                continue
            shutil.copy2(source, target)
        if conflicts:
            merge_conflicts.append({"subject": stem, "files": conflicts})
        else:
            required_present = all(
                (target_results / f"{stem}{suffix}").is_file()
                and (target_results / f"{stem}{suffix}").stat().st_size > 100
                for suffix in REQUIRED_SUFFIXES
            )
            if required_present:
                merged_subjects.append(stem)

    final_report_path = plan["side_root"] / "verify_spharm_outputs.json"
    final_report = inspect_retry(plan["input_dir"], plan["side_root"], final_report_path)
    return {
        "side": side,
        "attempted": len(plan["failed_names"]),
        "failed_names": plan["failed_names"],
        "template": str(plan["template"]),
        "template_sha256": plan["template_sha256"],
        "production_config": plan["config"],
        "runner_exit_code": runner_result.returncode,
        "retry_report": retry_report,
        "merged_subjects": merged_subjects,
        "merge_conflicts": merge_conflicts,
        "final_report": final_report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retry only failed SPHARM subjects using the original per-side reference templates."
    )
    parser.add_argument("--run-root", required=True, help="Existing legacy ICP/SPHARM run directory")
    parser.add_argument("--slicer-exe", default=str(DEFAULT_SLICER), help="SlicerSALT.exe path")
    parser.add_argument("--workers", type=int, default=1, help="Retry worker count; defaults to one for isolation")
    parser.add_argument("--check-only", action="store_true", help="Validate plan and hashes without running")
    args = parser.parse_args()

    run_root = Path(args.run_root).resolve()
    slicer_exe = Path(args.slicer_exe).resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(f"Run root does not exist: {run_root}")
    if not slicer_exe.is_file():
        raise FileNotFoundError(f"Slicer executable does not exist: {slicer_exe}")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    plans = [load_side_plan(run_root, side) for side in ("left", "right")]
    print(f"Run root: {run_root}")
    for plan in plans:
        print(
            f"{plan['side']}: retry {len(plan['failed_names'])} subject(s); "
            f"excluded {len(plan.get('excluded_names', []))}; "
            f"template={plan['template']}; SHA-256={plan['template_sha256']}"
        )
    if args.check_only:
        print("Preflight passed; no files were changed and SPHARM was not started.")
        return 0
    if not any(plan["failed_names"] for plan in plans):
        print("No failed SPHARM subjects were found.")
        return 0

    retry_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    retry_root = run_root / f"spharm_failed_retry_{retry_id}"
    if retry_root.exists():
        raise FileExistsError(f"Retry output already exists: {retry_root}")
    retry_root.mkdir(parents=True)

    results = []
    for plan in plans:
        results.append(run_side(plan, retry_root, slicer_exe, args.workers))

    manifest = {
        "schema": "spharm_failed_retry_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "parent_run_root": str(run_root),
        "retry_root": str(retry_root),
        "workers": args.workers,
        "results": results,
    }
    write_json(retry_root / "retry_manifest.json", manifest)

    failed_final = sum(
        result.get("final_report", {}).get("failed", 0)
        for result in results
        if result.get("final_report") is not None
    )
    for result in results:
        final = result.get("final_report")
        if final:
            print(
                f"{result['side']}: retried {result['attempted']}; "
                f"merged {len(result['merged_subjects'])}; "
                f"final complete {final['complete']}/{final['total_inputs']}; "
                f"remaining incomplete {final['failed']}"
            )
    print(f"Retry manifest: {retry_root / 'retry_manifest.json'}")
    return 1 if failed_final else 0


if __name__ == "__main__":
    raise SystemExit(main())

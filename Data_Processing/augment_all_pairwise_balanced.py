"""Create two deterministic, train-only pairwise augmentation protocols.

This script is intentionally separate from the older PLS-DA augmentation
scripts.  It consumes the already-built patient-level 80/20 ``All_coef``
split, never reads the test rows while fitting or pairing, and does not make a
validation split.  It writes one augmented training CSV and an untouched test
CSV for each side and protocol.

Protocols
---------
``between_class``
    Greedy nearest one-to-one matching between class 0 and class 1 in the
    standardized coefficient space.  A pair can produce at most eight
    children.  Four children are close to the class-0 endpoint (alpha
    0.10..0.40, label 0) and four are close to class 1 (alpha 0.60..0.90,
    label 1).  Thus every child's label is the class of its closest endpoint.

``within_class``
    Greedy nearest one-to-one matching within each class.  Each pair can
    produce at most eight children (alpha 0.10..0.90, label inherited from the
    pair's class).

The number of children is selected to make the final train classes equal while
maximising that equal class size subject to the one-use-per-pair and eight
children-per-pair constraints.  The scaler is fitted on the original training
rows only and saved as a sidecar for auditability.  PLS-DA is not applied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


META_COLUMNS = ["Subject", "Group", "Class", "BinaryClass", "DataType"]
FEATURE_COLUMNS = [f"Coef_{i}" for i in range(1, 508)]
ALL_SIDES = ("left", "right")
ALLOWED_PROTOCOLS = ("between_class", "within_class")

# Four children are close to each endpoint in the cross-class protocol.  None
# of these values is 0 or 1, so a child cannot silently duplicate a parent.
CROSS_ALPHAS = (0.10, 0.20, 0.30, 0.40, 0.60, 0.70, 0.80, 0.90)
WITHIN_ALPHAS = CROSS_ALPHAS


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_ready(value):
    """Convert numpy scalars/paths to values accepted by json.dumps."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def canonical_input(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected = META_COLUMNS + FEATURE_COLUMNS
    if frame.columns.tolist() != expected:
        raise ValueError(
            f"Unexpected schema in {path}: expected {len(expected)} columns, "
            f"got {len(frame.columns)}"
        )
    if frame["Subject"].isna().any() or frame["Subject"].astype(str).duplicated().any():
        raise ValueError(f"Subject is missing or duplicated in {path}")
    labels = pd.to_numeric(frame["BinaryClass"], errors="coerce")
    classes = pd.to_numeric(frame["Class"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError(f"BinaryClass must contain only 0/1: {path}")
    if classes.isna().any() or not classes.isin([0, 1]).all():
        raise ValueError(f"Class must contain only 0/1 for this binary protocol: {path}")
    if not np.array_equal(labels.to_numpy(dtype=int), classes.to_numpy(dtype=int)):
        raise ValueError(f"Class/BinaryClass disagree: {path}")
    if not frame["DataType"].fillna("Original").astype(str).str.lower().isin(
        ["original", "real"]
    ).all():
        raise ValueError(f"Input contains non-original rows: {path}")
    values = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"Input contains non-finite coefficients: {path}")
    result = frame.copy()
    result["Class"] = classes.astype(np.int64)
    result["BinaryClass"] = labels.astype(np.int64)
    result["DataType"] = "Original"
    return result


def fit_scaler(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Fit a StandardScaler-equivalent using train rows only."""
    mean = np.mean(values, axis=0, dtype=np.float64)
    scale = np.std(values, axis=0, dtype=np.float64)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return mean, scale


def standardize(values: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
    return (values - mean) / scale


def pair_distance(a: np.ndarray, b: np.ndarray) -> float:
    delta = a - b
    return float(np.sqrt(np.dot(delta, delta)))


def greedy_cross_pairs(z0: np.ndarray, z1: np.ndarray) -> List[Tuple[int, int, float]]:
    """Greedy globally-nearest one-to-one class-0/class-1 matching."""
    if len(z0) == 0 or len(z1) == 0:
        return []
    distances = np.sqrt(
        np.maximum(
            0.0,
            np.sum((z0[:, None, :] - z1[None, :, :]) ** 2, axis=2, dtype=np.float64),
        )
    )
    unused0 = np.ones(len(z0), dtype=bool)
    unused1 = np.ones(len(z1), dtype=bool)
    pairs: List[Tuple[int, int, float]] = []
    for _ in range(min(len(z0), len(z1))):
        masked = np.where(unused0[:, None] & unused1[None, :], distances, np.inf)
        flat = int(np.argmin(masked))
        i0, i1 = np.unravel_index(flat, masked.shape)
        distance = float(masked[i0, i1])
        if not np.isfinite(distance):
            break
        unused0[i0] = False
        unused1[i1] = False
        pairs.append((int(i0), int(i1), distance))
    return pairs


def greedy_within_pairs(z: np.ndarray) -> List[Tuple[int, int, float]]:
    """Greedy globally-nearest one-to-one matching within one class."""
    n = len(z)
    if n < 2:
        return []
    distances = np.sqrt(
        np.maximum(
            0.0,
            np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=2, dtype=np.float64),
        )
    )
    distances[np.diag_indices_from(distances)] = np.inf
    unused = np.ones(n, dtype=bool)
    pairs: List[Tuple[int, int, float]] = []
    for _ in range(n // 2):
        masked = np.where(unused[:, None] & unused[None, :], distances, np.inf)
        flat = int(np.argmin(masked))
        first, second = np.unravel_index(flat, masked.shape)
        distance = float(masked[first, second])
        if not np.isfinite(distance):
            break
        unused[first] = False
        unused[second] = False
        pairs.append((int(first), int(second), distance))
    return pairs


def mode_group(frame: pd.DataFrame, label: int) -> str:
    values = frame.loc[frame["BinaryClass"] == label, "Group"].astype(str)
    if values.empty:
        return f"Class_{label}"
    return str(Counter(values).most_common(1)[0][0])


def child_row(
    *,
    protocol: str,
    side: str,
    pair_number: int,
    child_number: int,
    alpha: float,
    label: int,
    x_child: np.ndarray,
    group: str,
) -> dict:
    row = {
        "Subject": f"aug_{protocol}_{side}_pair{pair_number:04d}_child{child_number:02d}",
        "Group": group,
        "Class": int(label),
        "BinaryClass": int(label),
        "DataType": "Augmented",
    }
    row.update({column: float(value) for column, value in zip(FEATURE_COLUMNS, x_child)})
    return row


def create_augmented(
    train: pd.DataFrame,
    side: str,
    protocol: str,
    children_per_pair: int,
    mean: np.ndarray,
    scale: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    if children_per_pair != 8:
        raise ValueError("This protocol is defined for exactly 8 children per pair")

    values = train[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    z = standardize(values, mean, scale)
    class_indices = {
        label: np.flatnonzero(train["BinaryClass"].to_numpy(dtype=int) == label)
        for label in (0, 1)
    }
    counts = {label: int(len(class_indices[label])) for label in (0, 1)}
    groups = {label: mode_group(train, label) for label in (0, 1)}
    pair_records: List[dict] = []
    synthetic_rows: List[dict] = []
    synthetic_manifest: List[dict] = []

    if protocol == "between_class":
        local_pairs = greedy_cross_pairs(z[class_indices[0]], z[class_indices[1]])
        pair_capacity = {0: 4 * len(local_pairs), 1: 4 * len(local_pairs)}
        target = min(counts[0] + pair_capacity[0], counts[1] + pair_capacity[1])
        needed = {label: target - counts[label] for label in (0, 1)}
        for pair_no, (local0, local1, distance) in enumerate(local_pairs, start=1):
            global0 = int(class_indices[0][local0])
            global1 = int(class_indices[1][local1])
            subject0 = str(train.iloc[global0]["Subject"])
            subject1 = str(train.iloc[global1]["Subject"])
            generated = 0
            generated_labels: List[int] = []
            for child_no, alpha in enumerate(CROSS_ALPHAS, start=1):
                label = 0 if alpha < 0.5 else 1
                if needed[label] <= 0:
                    continue
                z_child = (1.0 - alpha) * z[global0] + alpha * z[global1]
                x_child = z_child * scale + mean
                synthetic_rows.append(
                    child_row(
                        protocol=protocol,
                        side=side,
                        pair_number=pair_no,
                        child_number=child_no,
                        alpha=alpha,
                        label=label,
                        x_child=x_child,
                        group=groups[label],
                    )
                )
                synthetic_manifest.append(
                    {
                        "PairID": f"{protocol}_{side}_pair{pair_no:04d}",
                        "pair_number": pair_no,
                        "child_number": child_no,
                        "alpha": alpha,
                        "child_class": label,
                        "closest_endpoint_class": label,
                        "parent_subject_class_0": subject0,
                        "parent_subject_class_1": subject1,
                        "standardized_parent_distance": distance,
                    }
                )
                needed[label] -= 1
                generated += 1
                generated_labels.append(label)
            if generated:
                pair_records.append(
                    {
                        "PairID": f"{protocol}_{side}_pair{pair_no:04d}",
                        "pair_number": pair_no,
                        "pair_type": "cross_class",
                        "parent_subject_class_0": subject0,
                        "parent_subject_class_1": subject1,
                        "parent_class_0": 0,
                        "parent_class_1": 1,
                        "standardized_parent_distance": distance,
                        "children_generated": generated,
                        "children_per_pair_limit": children_per_pair,
                        "generated_child_classes": ",".join(map(str, generated_labels)),
                    }
                )
    elif protocol == "within_class":
        all_pairs: Dict[int, List[Tuple[int, int, float]]] = {}
        capacity = {}
        for label in (0, 1):
            local = greedy_within_pairs(z[class_indices[label]])
            all_pairs[label] = local
            capacity[label] = children_per_pair * len(local)
        target = min(counts[0] + capacity[0], counts[1] + capacity[1])
        needed = {label: target - counts[label] for label in (0, 1)}
        pair_no = 0
        for label in (0, 1):
            for local_first, local_second, distance in all_pairs[label]:
                if needed[label] <= 0:
                    break
                pair_no += 1
                global_first = int(class_indices[label][local_first])
                global_second = int(class_indices[label][local_second])
                subject_first = str(train.iloc[global_first]["Subject"])
                subject_second = str(train.iloc[global_second]["Subject"])
                generated = 0
                generated_labels: List[int] = []
                for child_no, alpha in enumerate(WITHIN_ALPHAS, start=1):
                    if needed[label] <= 0:
                        break
                    z_child = (1.0 - alpha) * z[global_first] + alpha * z[global_second]
                    x_child = z_child * scale + mean
                    synthetic_rows.append(
                        child_row(
                            protocol=protocol,
                            side=side,
                            pair_number=pair_no,
                            child_number=child_no,
                            alpha=alpha,
                            label=label,
                            x_child=x_child,
                            group=groups[label],
                        )
                    )
                    synthetic_manifest.append(
                        {
                            "PairID": f"{protocol}_{side}_pair{pair_no:04d}",
                            "pair_number": pair_no,
                            "child_number": child_no,
                            "alpha": alpha,
                            "child_class": label,
                            "closest_endpoint_class": label,
                            "parent_subject_class_0": subject_first if label == 0 else "",
                            "parent_subject_class_1": subject_first if label == 1 else "",
                            "parent_subject_same_class_2": subject_second,
                            "standardized_parent_distance": distance,
                        }
                    )
                    needed[label] -= 1
                    generated += 1
                    generated_labels.append(label)
                pair_records.append(
                    {
                        "PairID": f"{protocol}_{side}_pair{pair_no:04d}",
                        "pair_number": pair_no,
                        "pair_type": "within_class",
                        "parent_subject_class": label,
                        "parent_subject_1": subject_first,
                        "parent_subject_2": subject_second,
                        "parent_class": label,
                        "standardized_parent_distance": distance,
                        "children_generated": generated,
                        "children_per_pair_limit": children_per_pair,
                        "generated_child_classes": ",".join(map(str, generated_labels)),
                    }
                )
    else:
        raise ValueError(f"Unknown protocol: {protocol}")

    if needed != {0: 0, 1: 0}:
        raise AssertionError(f"Could not reach balanced target; remaining={needed}")
    synthetic = pd.DataFrame(synthetic_rows, columns=META_COLUMNS + FEATURE_COLUMNS)
    augmented = pd.concat([train, synthetic], ignore_index=True)
    # Stable order makes reruns byte-for-byte reproducible apart from metadata
    # timestamps in the JSON manifest.
    augmented = augmented.sort_values("Subject", kind="stable").reset_index(drop=True)
    pair_frame = pd.DataFrame(pair_records)
    synthetic_frame = pd.DataFrame(synthetic_manifest)
    if pair_frame["PairID"].duplicated().any():
        raise AssertionError("A pair was used more than once")
    if len(synthetic_frame) != len(synthetic):
        raise AssertionError("Synthetic manifest and synthetic CSV differ")
    final_counts = {
        str(label): int((augmented["BinaryClass"] == label).sum()) for label in (0, 1)
    }
    details = {
        "original_train_rows": len(train),
        "original_class_0": counts[0],
        "original_class_1": counts[1],
        "pair_count": len(pair_frame),
        "synthetic_rows": len(synthetic),
        "synthetic_class_0": int((synthetic["BinaryClass"] == 0).sum()),
        "synthetic_class_1": int((synthetic["BinaryClass"] == 1).sum()),
        "target_final_class_size": int(target),
        "final_train_rows": len(augmented),
        "final_class_0": final_counts["0"],
        "final_class_1": final_counts["1"],
        "unique_pair_ids": int(pair_frame["PairID"].nunique()),
        "pair_ids_reused": False,
        "children_per_pair_limit": children_per_pair,
    }
    return augmented, synthetic, pair_frame, {**details, "synthetic_manifest": synthetic_frame}


def save_side_protocol(
    *,
    input_root: Path,
    output_root: Path,
    side: str,
    protocol: str,
    seed: int,
    children_per_pair: int,
) -> dict:
    side_cap = side.capitalize()
    input_side = input_root / "All_coef" / side
    train_path = input_side / f"All_coef_{side_cap}_train_coef_features.csv"
    test_path = input_side / f"All_coef_{side_cap}_test_coef_features.csv"
    train = canonical_input(train_path)
    test = canonical_input(test_path)
    train_subjects = set(train["Subject"].astype(str))
    test_subjects = set(test["Subject"].astype(str))
    if train_subjects & test_subjects:
        raise ValueError(f"Train/test Subject overlap for {side}: {sorted(train_subjects & test_subjects)[:3]}")
    train_values = train[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    mean, scale = fit_scaler(train_values)
    augmented, synthetic, pair_frame, details = create_augmented(
        train=train,
        side=side,
        protocol=protocol,
        children_per_pair=children_per_pair,
        mean=mean,
        scale=scale,
    )

    side_root = output_root / protocol / side
    train_dir = side_root / "train"
    test_dir = side_root / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    out_train = train_dir / train_path.name
    out_test = test_dir / test_path.name
    out_synthetic = side_root / "synthetic_rows.csv"
    out_synthetic_manifest = side_root / "synthetic_manifest.csv"
    out_pairs = side_root / "pair_manifest.csv"
    out_scaler = side_root / "scaler_stats.csv"

    augmented.to_csv(out_train, index=False, float_format="%.17g")
    # Copy the untouched test bytes instead of reserialising the test DataFrame.
    shutil.copy2(test_path, out_test)
    synthetic.to_csv(out_synthetic, index=False, float_format="%.17g")
    details["synthetic_manifest"].to_csv(out_synthetic_manifest, index=False)
    pair_frame.to_csv(out_pairs, index=False)
    pd.DataFrame({"feature": FEATURE_COLUMNS, "mean": mean, "scale": scale}).to_csv(
        out_scaler, index=False, float_format="%.17g"
    )

    input_sidecar = Path(str(train_path) + ".json")
    input_manifest = json.loads(input_sidecar.read_text(encoding="utf-8")) if input_sidecar.is_file() else {}
    base_contract = input_manifest.get("feature_contract", {})
    common = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "pairwise-balanced-train-only-v1",
        "augmentation_protocol": protocol,
        "dataset": "All_coef",
        "side": side,
        "seed": int(seed),
        "children_per_pair": int(children_per_pair),
        "validation_rows": 0,
        "validation_created": False,
        "pls_da_applied_to_model_input": False,
        "pairing_space": "507-D coefficient space after train-only standardization",
        "scaler_fit_scope": "original training rows only",
        "train_only_augmentation": True,
        "test_untouched": True,
        "feature_contract": base_contract,
    }
    train_manifest = {
        **common,
        "split": "train",
        "input_train_csv": train_path.resolve(),
        "output_csv": out_train.resolve(),
        "input_train_sha256": sha256_file(train_path),
        "output_sha256": sha256_file(out_train),
        "original_train_rows": len(train),
        "output_train_rows": len(augmented),
        "synthetic_rows": len(synthetic),
        "class_counts_original": {str(k): int(v) for k, v in train["BinaryClass"].value_counts().sort_index().items()},
        "class_counts_output": {str(k): int(v) for k, v in augmented["BinaryClass"].value_counts().sort_index().items()},
        "pair_count": int(details["pair_count"]),
        "pair_ids_reused": False,
        "artifacts": {
            "synthetic_rows": out_synthetic.resolve(),
            "synthetic_manifest": out_synthetic_manifest.resolve(),
            "pair_manifest": out_pairs.resolve(),
            "scaler_stats": out_scaler.resolve(),
        },
    }
    test_manifest = {
        **common,
        "split": "test",
        "input_test_csv": test_path.resolve(),
        "output_csv": out_test.resolve(),
        "input_test_sha256": sha256_file(test_path),
        "output_sha256": sha256_file(out_test),
        "rows": len(test),
        "synthetic_rows": 0,
    }
    write_json(Path(str(out_train) + ".json"), train_manifest)
    write_json(Path(str(out_test) + ".json"), test_manifest)
    details_for_summary = {k: v for k, v in details.items() if k != "synthetic_manifest"}
    return {
        "protocol": protocol,
        "side": side,
        "input_train": str(train_path.resolve()),
        "input_test": str(test_path.resolve()),
        "train_csv": str(out_train.resolve()),
        "test_csv": str(out_test.resolve()),
        "synthetic_csv": str(out_synthetic.resolve()),
        "pair_manifest": str(out_pairs.resolve()),
        "scaler_stats": str(out_scaler.resolve()),
        "test_sha256_unchanged": sha256_file(test_path) == sha256_file(out_test),
        **details_for_summary,
    }


def make_readme(root: Path, summary: pd.DataFrame, input_root: Path, seed: int) -> None:
    lines = [
        "# All_coef pairwise augmentation (80/20, no validation)",
        "",
        "สร้างเมื่อ: " + datetime.now().astimezone().isoformat(timespec="seconds"),
        "",
        "ชุดนี้ใช้เฉพาะ `All_coef` ซ้าย/ขวา จาก split 80/20 ที่มีอยู่แล้ว ไม่สร้าง validation และ augment เฉพาะ train เท่านั้น",
        "test ถูกคัดลอก byte ต่อ byte จาก input และไม่ถูกใช้คำนวณ scaler, จับคู่ หรือสร้างข้อมูลสังเคราะห์",
        "",
        "## โปรโตคอล",
        "",
        "- `between_class`: จับคู่ class 0 กับ class 1 ที่ใกล้กันที่สุดแบบหนึ่งต่อหนึ่งใน coefficient space หลัง standardize ด้วย train เท่านั้น; คู่หนึ่งสร้างได้ไม่เกิน 8 แถว: alpha 0.10, 0.20, 0.30, 0.40 ติดป้าย class 0 และ alpha 0.60, 0.70, 0.80, 0.90 ติดป้าย class 1 ซึ่งเป็นคลาสของ endpoint ที่ใกล้กว่า",
        "- `within_class`: จับคู่เฉพาะแถวใน class เดียวกันแบบหนึ่งต่อหนึ่ง; คู่หนึ่งสร้างได้ไม่เกิน 8 แถวและรับ label ของคลาสนั้น",
        "- ไม่ใช้ PLS-DA และไม่ใช้คู่เดิมซ้ำ; จำนวนแถวถูกเลือกให้ได้ขนาดสองคลาสเท่ากันสูงสุดภายใต้ข้อจำกัดดังกล่าว",
        "",
        "## โฟลเดอร์ที่ใช้เทรนและเทส",
        "",
        "| protocol | side | train (ใช้เทรน) | test (ใช้ประเมินครั้งสุดท้าย) |",
        "|---|---|---|---|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(f"| `{row.protocol}` | `{row.side}` | `{row.train_csv}` | `{row.test_csv}` |")
    lines += [
        "",
        "## จำนวนที่ได้",
        "",
        "| protocol | side | original train (0/1) | synthetic (0/1) | final train (0/1) | pairs |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| `{row.protocol}` | `{row.side}` | {row.original_class_0}/{row.original_class_1} "
            f"| {row.synthetic_class_0}/{row.synthetic_class_1} | {row.final_class_0}/{row.final_class_1} "
            f"| {row.pair_count} |"
        )
    lines += [
        "",
        "ไฟล์ตรวจสอบในแต่ละ side: `pair_manifest.csv` (หนึ่งแถวต่อคู่และไม่ซ้ำ), `synthetic_manifest.csv` (หนึ่งแถวต่อลูก), `scaler_stats.csv` และ `augmentation_manifest.json` ที่อยู่ข้าง CSV train/test",
        "",
        "คำสั่ง rerun:",
        "```powershell",
        f"$env:PYTHONPATH=\"$PWD\\Model\\_training_cuda_site;$PWD\\Model\\_training_site\"",
        f"& 'C:\\Program Files\\SlicerSALT 6.0.0\\bin\\PythonSlicer.exe' Data_Processing\\augment_all_pairwise_balanced.py --data-root \"{input_root}\" --output-root \"{root}\" --protocol all --side all --children-per-pair 8 --seed {seed}",
        "```",
        "",
        "คำเตือนสำหรับการใช้ในงานวิจัย: ชุด train นี้เป็น pre-augmented fixed-split สำหรับการทดลองแบบ 80/20 ตามคำขอ หากทำ cross-validation ต้องย้ายการจับคู่เข้าไปภายในแต่ละ training fold เพื่อไม่ให้ข้อมูลสังเคราะห์รั่วข้าม fold",
    ]
    (root / "README_TH.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True, help="Model/current_spharm_80_20_20260922")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", choices=["between_class", "within_class", "all"], default="all")
    parser.add_argument("--side", choices=["left", "right", "all"], default="all")
    parser.add_argument("--children-per-pair", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    sides = ALL_SIDES if args.side == "all" else (args.side,)
    protocols = ALLOWED_PROTOCOLS if args.protocol == "all" else (args.protocol,)
    if args.children_per_pair != 8:
        raise SystemExit("--children-per-pair must be exactly 8 for this protocol")
    if not (input_root / "All_coef").is_dir():
        raise SystemExit(f"Missing input All_coef directory: {input_root / 'All_coef'}")
    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for protocol in protocols:
        for side in sides:
            summary_rows.append(
                save_side_protocol(
                    input_root=input_root,
                    output_root=output_root,
                    side=side,
                    protocol=protocol,
                    seed=args.seed,
                    children_per_pair=args.children_per_pair,
                )
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_root / "SUMMARY.csv", index=False)
    run_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": Path(__file__).resolve(),
        "input_root": input_root,
        "output_root": output_root,
        "dataset": "All_coef",
        "sides": list(sides),
        "protocols": list(protocols),
        "seed": args.seed,
        "children_per_pair": args.children_per_pair,
        "validation_created": False,
        "augmentation_only_on_train": True,
        "pls_da_used": False,
        "summary_csv": output_root / "SUMMARY.csv",
        "readme": output_root / "README_TH.md",
        "runs": summary_rows,
    }
    write_json(output_root / "RUN_MANIFEST.json", run_manifest)
    make_readme(output_root, summary, input_root, args.seed)
    print(summary.to_string(index=False))
    print(f"SUMMARY: {output_root / 'SUMMARY.csv'}")
    print(f"README: {output_root / 'README_TH.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

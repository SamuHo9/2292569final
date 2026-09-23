"""Report the exact fold-local PLS-DA/augmentation counts for the current split."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from data_contract import load_raw_cohort, patient_group_id
from leakage_free_coef_training import grouped_splits


COHORTS = ("All_coef", "All_coef_Ds004469", "All_coef_Ds005602")
SIDES = ("left", "right")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pls_components", type=int, default=8)
    args = parser.parse_args()

    rows = []
    for cohort in COHORTS:
        for side in SIDES:
            train, test = load_raw_cohort(cohort, side, kind="coef", root=args.data_root)
            y = train.BinaryClass.to_numpy(dtype="int64")
            import numpy as np
            groups = np.asarray([patient_group_id(value) for value in train.Subject], dtype=str)
            splits = grouped_splits(y, groups, args.folds, args.seed)
            fold_synthetic = []
            for fold_index, (tr, va) in enumerate(splits, start=1):
                train_counts = {label: int((y[tr] == label).sum()) for label in (0, 1)}
                val_counts = {label: int((y[va] == label).sum()) for label in (0, 1)}
                synthetic = abs(train_counts[0] - train_counts[1])
                fold_synthetic.append(synthetic)
                rows.append({
                    "Cohort": cohort,
                    "Side": side,
                    "Fold": fold_index,
                    "Train_rows_before_aug": int(len(tr)),
                    "Fold_train_class_0": train_counts[0],
                    "Fold_train_class_1": train_counts[1],
                    "Synthetic_rows_for_balance": synthetic,
                    "Fold_val_rows": int(len(va)),
                    "Fold_val_class_0": val_counts[0],
                    "Fold_val_class_1": val_counts[1],
                    "PLS_components_requested": int(args.pls_components),
                    "PLS_components_actual": int(min(args.pls_components, 507, len(tr) - 1)),
                    "Train_total_rows": int(len(train)),
                    "Test_total_rows": int(len(test)),
                    "Train_class_0": int((y == 0).sum()),
                    "Train_class_1": int((y == 1).sum()),
                })
            print(
                f"{cohort}/{side}: train={len(train)} test={len(test)} "
                f"synthetic_per_fold={fold_synthetic} total={sum(fold_synthetic)}"
            )
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()

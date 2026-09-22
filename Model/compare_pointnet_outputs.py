"""Compare raw XYZ PointNet with fold-local PLS-DA PointNet outputs."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def summarize(root: Path, label: str):
    frame = pd.read_csv(root / "OPTUNA_FINAL_SUMMARY.csv")
    grouped = frame.groupby(["Cohort", "Side"], as_index=False)
    out = grouped.agg(
        OOF_BA=("BestOOF_BalancedAccuracy", "first"),
        Test_Accuracy_Mean=("Test_Accuracy", "mean"),
        Test_Accuracy_SD=("Test_Accuracy", "std"),
        Test_BA_Mean=("Test_BalancedAccuracy", "mean"),
        Test_BA_SD=("Test_BalancedAccuracy", "std"),
        Test_F1_Macro_Mean=("Test_F1_Macro", "mean"),
        Test_ROC_AUC_Mean=("Test_ROC_AUC", "mean"),
    )
    out.insert(2, "Protocol", label)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_root", required=True, type=Path)
    parser.add_argument("--plsda_root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raw = summarize(args.raw_root.resolve(), "PointNet raw XYZ")
    plsda = summarize(args.plsda_root.resolve(), "PointNet PLS-DA")
    combined = pd.concat([raw, plsda], ignore_index=True)
    pivot = combined.pivot(index=["Cohort", "Side"], columns="Protocol")
    pivot.columns = [f"{metric}_{protocol.replace(' ', '_')}" for metric, protocol in pivot.columns]
    pivot = pivot.reset_index()
    for metric in ("Test_Accuracy_Mean", "Test_BA_Mean", "Test_F1_Macro_Mean", "Test_ROC_AUC_Mean"):
        raw_col = f"{metric}_PointNet_raw_XYZ"
        plsda_col = f"{metric}_PointNet_PLS-DA"
        pivot[f"Delta_{metric}_PLSDA_minus_raw"] = pivot[plsda_col] - pivot[raw_col]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pivot.to_csv(args.output, index=False)
    print(f"Wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()

"""Tune decision thresholds for an existing fixed-topology Optuna run.

The script regenerates training-only grouped OOF predictions using the best
Optuna parameters already saved in ``OPTUNA_TUNING_SUMMARY.csv``.  It then
chooses a threshold from those OOF predictions and applies it to the already
saved test probabilities.  No test probability is used to choose a threshold
and the original metrics/predictions are left untouched.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from data_contract import load_raw_cohort
from leakage_free_coef_training import (
    COHORTS,
    MODELS,
    SIDES,
    grouped_splits,
    matrix,
    metrics,
    seed_everything,
    tensor_for_model,
    train_torch_fold,
)
from optuna_leakage_free_all import coef_training_fold, svm_model


def choose_threshold(y, probability):
    candidates = np.linspace(0.10, 0.90, 161)
    scored = []
    for threshold in candidates:
        result = metrics(y, probability, float(threshold))
        scored.append((result["balanced_accuracy"], result["f1_macro"],
                       -abs(float(threshold) - 0.5), float(threshold)))
    return max(scored)[-1]


def oof_for_best(cohort, side, model_name, params, args):
    train_df, _ = load_raw_cohort(cohort, side, kind="coef", root=args.data_root)
    x, y, groups, _ = matrix(train_df)
    splits = grouped_splits(y, groups, args.folds, args.seed)
    oof = np.zeros(len(y), dtype=float)
    for fold, (tr, va) in enumerate(splits):
        fold_seed = int(args.seed + fold)
        fit_x, fit_y, _ = coef_training_fold(
            x[tr], y[tr], fold_seed, args.protocol,
            args.children_per_pair, args.noise_scale, args.pls_components,
        )
        if model_name == "SVM":
            model = svm_model(params, fold_seed).fit(fit_x, fit_y)
            oof[va] = model.predict_proba(x[va])[:, 1]
        else:
            scaler = StandardScaler().fit(fit_x)
            model, val_prob, _ = train_torch_fold(
                model_name,
                scaler.transform(fit_x).astype(np.float32), fit_y,
                scaler.transform(x[va]).astype(np.float32), y[va],
                args.device_obj, fold_seed, args.tune_epochs, args.tune_patience,
                lr=float(params.get("lr", 1e-3)),
                weight_decay=float(params.get("weight_decay", 1e-3)),
                batch_size=int(params.get("batch_size", 32)),
                recon_weight=float(params.get("recon_weight", 0.05)),
                dropout=float(params.get("dropout", 0.25)),
            )
            oof[va] = val_prob
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    return train_df, groups, y, oof


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--protocol", choices=("coef_raw", "coef_plsda"), default="coef_plsda")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tune_epochs", type=int, default=20)
    parser.add_argument("--tune_patience", type=int, default=5)
    parser.add_argument("--children_per_pair", type=int, default=8)
    parser.add_argument("--noise_scale", type=float, default=0.02)
    parser.add_argument("--pls_components", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--reuse_thresholds", action="store_true",
                        help="Reuse FIXED_THRESHOLD_SUMMARY.csv and skip OOF regeneration")
    args = parser.parse_args()
    args.data_root = str(Path(args.data_root).resolve())
    args.run_root = Path(args.run_root).resolve()
    args.device_obj = torch.device(
        args.device if args.device != "auto" else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    tuning_path = args.run_root / "OPTUNA_TUNING_SUMMARY.csv"
    final_path = args.run_root / "OPTUNA_FINAL_SUMMARY.csv"
    tuning = pd.read_csv(tuning_path)
    final = pd.read_csv(final_path)
    threshold_rows = []
    final_rows = []
    oof_root = args.run_root / "threshold_tuning"
    oof_root.mkdir(parents=True, exist_ok=True)

    existing_thresholds = args.run_root / "FIXED_THRESHOLD_SUMMARY.csv"
    if args.reuse_thresholds and existing_thresholds.is_file():
        threshold_rows = pd.read_csv(existing_thresholds).to_dict("records")
    else:
        for row in tuning.itertuples(index=False):
            params = json.loads(row.BestParams)
            train_df, groups, y, oof = oof_for_best(
                row.Cohort, row.Side, row.Model, params, args
            )
            threshold = choose_threshold(y, oof)
            group_dir = oof_root / row.Cohort / row.Side / row.Model
            group_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({
                "Subject": train_df["Subject"].astype(str),
                "PatientID": groups,
                "BinaryClass": y,
                "Probability": oof,
                "Prediction_0_5": (oof >= 0.5).astype(int),
                "Prediction": (oof >= threshold).astype(int),
            }).to_csv(group_dir / "best_oof_predictions.csv", index=False)
            oof_default = metrics(y, oof, 0.5)
            oof_tuned = metrics(y, oof, threshold)
            threshold_rows.append({
                "Protocol": row.Protocol,
                "Cohort": row.Cohort,
                "Side": row.Side,
                "Model": row.Model,
                "Threshold": float(threshold),
                "OOF_BalancedAccuracy_0_5": oof_default["balanced_accuracy"],
                "OOF_BalancedAccuracy_Tuned": oof_tuned["balanced_accuracy"],
                "OOF_F1_Macro_Tuned": oof_tuned["f1_macro"],
                "OOF_Sensitivity_Tuned": oof_tuned["sensitivity"],
                "OOF_Specificity_Tuned": oof_tuned["specificity"],
                "ThresholdSource": "training_oof_only",
            })

    threshold_df = pd.DataFrame(threshold_rows)
    threshold_df.to_csv(args.run_root / "FIXED_THRESHOLD_SUMMARY.csv", index=False)
    lookup = {(r["Cohort"], r["Side"], r["Model"]): r for r in threshold_rows}
    for row in final.itertuples(index=False):
        key = (row.Cohort, row.Side, row.Model)
        threshold = float(lookup[key]["Threshold"])
        pred_path = Path(row.RunDir) / "test_predictions.csv"
        pred = pd.read_csv(pred_path)
        tuned_pred = (pred["Probability"].to_numpy(dtype=float) >= threshold).astype(int)
        test_y = pred["BinaryClass"].to_numpy(dtype=int)
        tuned = metrics(test_y, pred["Probability"].to_numpy(dtype=float), threshold)
        pred["Prediction_0_5"] = (pred["Probability"].to_numpy(dtype=float) >= 0.5).astype(int)
        pred["Prediction"] = tuned_pred
        pred.to_csv(Path(row.RunDir) / "test_predictions_threshold_tuned.csv", index=False)
        (Path(row.RunDir) / "metrics_threshold_tuned.json").write_text(
            json.dumps({"threshold": threshold, "test": tuned}, indent=2),
            encoding="utf-8",
        )
        values = row._asdict()
        values.update({
            "Threshold_OOF": threshold,
            "Test_Accuracy_Tuned": tuned["accuracy"],
            "Test_BalancedAccuracy_Tuned": tuned["balanced_accuracy"],
            "Test_Sensitivity_Tuned": tuned["sensitivity"],
            "Test_Specificity_Tuned": tuned["specificity"],
            "Test_F1_Macro_Tuned": tuned["f1_macro"],
            "Test_ROC_AUC_Tuned": tuned["roc_auc"],
            "Test_PR_AUC_Tuned": tuned["pr_auc"],
        })
        final_rows.append(values)
    pd.DataFrame(final_rows).to_csv(
        args.run_root / "OPTUNA_FINAL_THRESHOLD_TUNED.csv", index=False
    )
    (args.run_root / "FIXED_THRESHOLD_MANIFEST.json").write_text(json.dumps({
        "source_run_root": str(args.run_root),
        "data_root": args.data_root,
        "protocol": args.protocol,
        "folds": args.folds,
        "seed": args.seed,
        "threshold_source": "training_oof_only",
        "test_used_to_choose_threshold": False,
        "groups": len(COHORTS) * len(SIDES),
    }, indent=2), encoding="utf-8")
    print(f"Wrote {args.run_root / 'FIXED_THRESHOLD_SUMMARY.csv'}")
    print(f"Wrote {args.run_root / 'OPTUNA_FINAL_THRESHOLD_TUNED.csv'}")


if __name__ == "__main__":
    main()

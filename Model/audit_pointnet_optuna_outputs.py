"""Audit Optuna PointNet outputs without using test data for selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from data_contract import load_raw_xyz_cohort


COHORTS = ("All_coef", "All_coef_Ds004469", "All_coef_Ds005602")
SIDES = ("left", "right")
SEEDS = (42, 123, 2026)


def audit(root: Path, protocol: str, data_root: Path):
    summary_path = root / "OPTUNA_FINAL_SUMMARY.csv"
    tuning_path = root / "OPTUNA_TUNING_SUMMARY.csv"
    manifest_path = root / "RUN_MANIFEST.json"
    summary = pd.read_csv(summary_path)
    tuning = pd.read_csv(tuning_path)
    run_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {(c, s, "PointNet") for c in COHORTS for s in SIDES}
    assert len(tuning) == 6, f"Expected 6 studies, got {len(tuning)}"
    assert {(r.Cohort, r.Side, r.Model) for r in tuning.itertuples()} == expected
    assert len(summary) == 18, f"Expected 18 seed rows, got {len(summary)}"
    assert {(r.Cohort, r.Side, r.Model) for r in summary.itertuples()} == expected
    assert set(summary.EvalSeed.astype(int)) == set(SEEDS)
    assert not run_manifest.get("test_used_during_tuning", True)
    assert run_manifest.get("failures", []) == []
    assert run_manifest.get("folds_requested") == 10
    assert run_manifest.get("trials_per_study") == 10

    checked = 0
    for row in summary.itertuples():
        run = Path(row.RunDir)
        manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
        metric = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
        pred = pd.read_csv(run / "test_predictions.csv")
        assert manifest["protocol"] == f"optuna-{protocol}-v1"
        assert manifest["model"] == "PointNet"
        assert manifest["feature_kind"] == "xyz"
        assert manifest["folds"] == 10
        assert manifest["optuna_trials"] == 10
        assert manifest["children_per_pair"] == 8
        assert manifest["test_evaluation"] == "single_final_model_no_fold_ensemble"
        assert manifest["augmentation_applied_to_test"] is False
        assert manifest["augmentation_applied_to_validation"] is False
        if protocol == "pointnet_plsda":
            assert manifest["pls_da_fit_scope"] == "training_data_only"
            assert manifest["pls_components_requested"] == 8
        assert (run / "model_state.pt").is_file()
        _, test = load_raw_xyz_cohort(row.Cohort, row.Side, root=data_root)
        assert len(pred) == len(test)
        assert set(pred.Subject) == set(test.Subject)
        assert pred.Probability.between(0, 1).all()
        for section in ("test",):
            for key in ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1_macro"):
                assert 0.0 <= float(metric[section][key]) <= 1.0
        checked += 1

    result = {
        "run_root": str(root.resolve()),
        "protocol": protocol,
        "studies": len(tuning),
        "final_seed_runs": checked,
        "test_used_during_tuning": False,
        "failures": 0,
        "test_balanced_accuracy_mean": float(summary.Test_BalancedAccuracy.mean()),
        "test_accuracy_mean": float(summary.Test_Accuracy.mean()),
    }
    output = root / "POINTNET_OPTUNA_AUDIT.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote audit: {output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", required=True, type=Path)
    parser.add_argument("--protocol", choices=("pointnet_raw", "pointnet_plsda"), required=True)
    parser.add_argument("--data_root", required=True, type=Path)
    args = parser.parse_args()
    audit(args.run_root.resolve(), args.protocol, args.data_root.resolve())


if __name__ == "__main__":
    main()

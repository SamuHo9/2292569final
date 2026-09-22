"""Audit the completed architecture-optimized coefficient run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


MODELS = {"SVM", "MLP", "ResNet", "ResNetAE", "MobileNet", "SqueezeNet"}
COHORTS = {"All_coef", "All_coef_Ds004469", "All_coef_Ds005602"}
SIDES = {"left", "right"}


def check_metric(value):
    if value is None or pd.isna(value):
        return
    assert 0.0 <= float(value) <= 1.0, value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    tuning = pd.read_csv(root / "ARCH_OPT_TUNING_SUMMARY.csv")
    final = pd.read_csv(root / "ARCH_OPT_FINAL_SUMMARY.csv")
    expected = {(c, s, m) for c in COHORTS for s in SIDES for m in MODELS}
    keys = {(r.Cohort, r.Side, r.Model) for r in tuning.itertuples()}
    assert len(tuning) == 36, len(tuning)
    assert keys == expected, (expected - keys, keys - expected)
    assert len(final) == 108, len(final)
    assert not tuning.astype(str).apply(lambda col: col.str.contains('FAIL', case=False).any()).any()
    assert not final.astype(str).apply(lambda col: col.str.contains('FAIL', case=False).any()).any()
    assert final.groupby(["Cohort", "Side", "Model"]).size().eq(3).all()

    for row in tuning.itertuples():
        study_dir = root / "studies" / row.Protocol / row.Cohort / row.Side / row.Model
        for name in ("best_params.json", "trials.csv", "threshold.json", "best_oof_predictions.csv"):
            assert (study_dir / name).is_file(), study_dir / name
        threshold = json.loads((study_dir / "threshold.json").read_text(encoding="utf-8"))
        assert threshold["source"] == "best_trial_training_oof_only"
        assert 0.10 <= float(threshold["threshold"]) <= 0.90
        oof = pd.read_csv(study_dir / "best_oof_predictions.csv")
        assert len(oof) > 0
        assert oof["Probability"].between(0, 1).all()

    for row in final.itertuples():
        run_dir = Path(row.RunDir)
        for name in ("metrics.json", "test_predictions.csv", "run_manifest.json"):
            assert (run_dir / name).is_file(), run_dir / name
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["test_used_during_tuning"] is False
        assert manifest["threshold_source"] == "training_oof_only"
        assert manifest["augmentation_applied_to_test"] is False
        pred = pd.read_csv(run_dir / "test_predictions.csv")
        assert len(pred) == int(manifest["test_rows"])
        assert pred["Probability"].between(0, 1).all()
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        for section in ("test_default_0_5", "test_tuned_oof_threshold"):
            for key in ("accuracy", "balanced_accuracy", "sensitivity", "specificity", "f1_macro"):
                check_metric(metrics[section][key])

    report = {
        "run_root": str(root),
        "studies": len(tuning),
        "final_seed_runs": len(final),
        "models": sorted(MODELS),
        "cohorts": sorted(COHORTS),
        "sides": sorted(SIDES),
        "test_used_during_tuning": False,
        "threshold_source": "training_oof_only",
        "failures": 0,
        "test_balanced_accuracy_tuned_mean": float(final["Test_BalancedAccuracy_Tuned"].mean()),
        "test_accuracy_tuned_mean": float(final["Test_Accuracy_Tuned"].mean()),
    }
    out = root / "ARCH_OPT_AUDIT.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

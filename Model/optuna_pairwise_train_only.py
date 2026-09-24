"""Optuna tuning for pairwise augmentation with train-only grouped CV.

The fixed pre-augmented files are useful for a fixed 80/20 baseline, but using
those synthetic rows directly in cross-validation can put a synthetic child in
one fold while its parents are in another.  This runner therefore starts from
the original All_coef training rows and recreates the requested pairwise
augmentation inside every training fold.  The validation fold contains only
original rows and the held-out test set is never read by an Optuna objective.

After each study, the best hyperparameters are used to fit a final model on
the complete original training cohort after one full pairwise augmentation.
The final model is evaluated on the untouched test CSV for seeds 42, 123 and
2026.  This is the research-safe Optuna protocol for the new datasets.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

MODEL_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT / "Data_Processing"))

from augment_all_pairwise_balanced import (  # noqa: E402
    FEATURE_COLUMNS,
    META_COLUMNS,
    canonical_input,
    create_augmented,
    fit_scaler,
)
from leakage_free_coef_training import (  # noqa: E402
    MODELS,
    grouped_splits,
    make_torch_model,
    matrix,
    metrics,
    seed_everything,
    tensor_for_model,
    train_torch_fold,
)
from data_contract import patient_group_id  # noqa: E402


PROTOCOLS = ("between_class", "within_class")
SIDES = ("left", "right")
COHORT = "All_coef"


def json_ready(value):
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


def write_json(path: Path, payload: dict):
    path.write_text(json.dumps(json_ready(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sigmoid(values):
    values = np.clip(np.asarray(values, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-values))


def source_pair(data_root: Path, side: str):
    cap = side.capitalize()
    folder = data_root / COHORT / side
    train_path = folder / f"{COHORT}_{cap}_train_coef_features.csv"
    test_path = folder / f"{COHORT}_{cap}_test_coef_features.csv"
    train = canonical_input(train_path)
    test = canonical_input(test_path)
    if set(train.Subject.astype(str)) & set(test.Subject.astype(str)):
        raise ValueError(f"Train/test overlap: {side}")
    return train, test, train_path, test_path


def choose_params(trial: optuna.Trial, model_name: str) -> dict:
    if model_name == "SVM":
        gamma_mode = trial.suggest_categorical("gamma_mode", ["scale", "auto", "numeric"])
        params = {"C": trial.suggest_float("C", 1e-2, 1e2, log=True), "gamma_mode": gamma_mode}
        if gamma_mode == "numeric":
            params["gamma"] = trial.suggest_float("gamma", 1e-5, 1e-1, log=True)
        return params
    params = {
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "dropout": trial.suggest_float("dropout", 0.10, 0.50),
    }
    if model_name == "ResNetAE":
        params["recon_weight"] = trial.suggest_float("recon_weight", 0.01, 0.20, log=True)
    return params


def svm_pipeline(params: dict, seed: int) -> Pipeline:
    gamma = params.get("gamma_mode", "scale")
    if gamma == "numeric":
        gamma = float(params["gamma"])
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(C=float(params["C"]), kernel="rbf", gamma=gamma,
                       class_weight="balanced", probability=False, random_state=seed)),
    ])


def fold_augmented(frame: pd.DataFrame, side: str, protocol: str, seed: int):
    values = frame[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    mean, scale = fit_scaler(values)
    augmented, _, _, _ = create_augmented(
        train=frame, side=side, protocol=protocol,
        children_per_pair=8, mean=mean, scale=scale,
    )
    return augmented


def objective_factory(
    *,
    trial: optuna.Trial,
    model_name: str,
    protocol: str,
    side: str,
    train: pd.DataFrame,
    splits: list[tuple[np.ndarray, np.ndarray]],
    device: torch.device,
    seed: int,
    tune_epochs: int,
    tune_patience: int,
) -> float:
    params = choose_params(trial, model_name)
    y_all = train["BinaryClass"].to_numpy(dtype=np.int64)
    oof = np.zeros(len(train), dtype=np.float64)
    for fold, (tr_idx, va_idx) in enumerate(splits):
        fold_seed = seed + trial.number * 1000 + fold
        fold_frame = train.iloc[tr_idx].reset_index(drop=True)
        augmented = fold_augmented(fold_frame, side, protocol, fold_seed)
        x_train = augmented[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        y_train = augmented["BinaryClass"].to_numpy(dtype=np.int64)
        x_val = train.iloc[va_idx][FEATURE_COLUMNS].to_numpy(dtype=np.float32)
        y_val = y_all[va_idx]
        if model_name == "SVM":
            model = svm_pipeline(params, fold_seed).fit(x_train, y_train)
            oof[va_idx] = sigmoid(model.decision_function(x_val))
        else:
            scaler = StandardScaler().fit(x_train)
            scaled_train = scaler.transform(x_train).astype(np.float32)
            scaled_val = scaler.transform(x_val).astype(np.float32)
            _, val_prob, _ = train_torch_fold(
                model_name, scaled_train, y_train, scaled_val, y_val,
                device, fold_seed, tune_epochs, tune_patience,
                lr=params["lr"], weight_decay=params["weight_decay"],
                batch_size=params["batch_size"],
                recon_weight=params.get("recon_weight", 0.05),
                dropout=params.get("dropout"),
            )
            oof[va_idx] = val_prob
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        score = metrics(y_all[va_idx], oof[va_idx])["balanced_accuracy"]
        trial.report(float(score), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(metrics(y_all, oof)["balanced_accuracy"])


def fit_final_neural(name, x_train, y_train, x_test, device, seed, epochs, params):
    seed_everything(seed)
    model = make_torch_model(name, dropout=params.get("dropout")).to(device)
    counts = np.bincount(y_train, minlength=2).astype(np.float32)
    pos_weight = torch.tensor(
        [max(1.0, float(counts[0])) / max(1.0, float(counts[1]))],
        dtype=torch.float32, device=device,
    )
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(params.get("lr", 1e-3)),
        weight_decay=float(params.get("weight_decay", 1e-3)),
    )
    tx = tensor_for_model(name, x_train)
    ty = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(tx, ty),
        batch_size=min(int(params.get("batch_size", 32)), len(ty)),
        shuffle=True, drop_last=False,
    )
    logs = []
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        row_count = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            if name == "ResNetAE":
                logits, reconstruction = model(bx)
                loss = criterion(logits, by) + float(params.get("recon_weight", 0.05)) * torch.nn.functional.mse_loss(
                    reconstruction, bx.flatten(1)
                )
            else:
                loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(by)
            row_count += len(by)
        logs.append({"epoch": epoch, "train_loss": loss_sum / max(1, row_count)})
    model.eval()
    with torch.no_grad():
        logits = model(tensor_for_model(name, x_test).to(device))
        if name == "ResNetAE":
            logits = logits[0]
        probability = torch.sigmoid(logits).flatten().cpu().numpy()
    return model, probability, logs


def train_final(
    *,
    protocol: str,
    side: str,
    model_name: str,
    params: dict,
    train: pd.DataFrame,
    test: pd.DataFrame,
    device: torch.device,
    seed: int,
    final_epochs: int,
    output_dir: Path,
    train_path: Path,
    test_path: Path,
) -> dict:
    started = time.time()
    augmented = fold_augmented(train, side, protocol, seed)
    x_train = augmented[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train = augmented["BinaryClass"].to_numpy(dtype=np.int64)
    x_test = test[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_test = test["BinaryClass"].to_numpy(dtype=np.int64)
    output_dir.mkdir(parents=True, exist_ok=True)
    if model_name == "SVM":
        model = svm_pipeline(params, seed).fit(x_train, y_train)
        probability = sigmoid(model.decision_function(x_test))
        with (output_dir / "model.pkl").open("wb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pd.DataFrame([{"epoch": 1, "status": "fit_complete"}]).to_csv(output_dir / "training_log.csv", index=False)
    else:
        scaler = StandardScaler().fit(x_train)
        scaled_train = scaler.transform(x_train).astype(np.float32)
        scaled_test = scaler.transform(x_test).astype(np.float32)
        model, probability, logs = fit_final_neural(
            model_name, scaled_train, y_train, scaled_test, device,
            seed, final_epochs, params,
        )
        torch.save(model.state_dict(), output_dir / "model_state.pt")
        with (output_dir / "scaler.pkl").open("wb") as handle:
            pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pd.DataFrame(logs).to_csv(output_dir / "training_log.csv", index=False)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    test_metrics = metrics(y_test, probability)
    pd.DataFrame({
        "Subject": test["Subject"].astype(str),
        "BinaryClass": y_test,
        "Probability": probability,
        "Prediction": (probability >= 0.5).astype(int),
    }).to_csv(output_dir / "test_predictions.csv", index=False)
    write_json(output_dir / "metrics.json", {"test": test_metrics})
    write_json(output_dir / "run_manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "optuna-pairwise-train-only-v1",
        "augmentation_protocol": protocol,
        "dataset": COHORT,
        "side": side,
        "model": model_name,
        "seed": seed,
        "train_csv": train_path.resolve(),
        "test_csv": test_path.resolve(),
        "train_rows_original": len(train),
        "train_rows_after_pairwise_augmentation": len(augmented),
        "test_rows": len(test),
        "validation_created_for_final_fit": False,
        "optuna_cv_used_for_selection": True,
        "pls_da_applied_to_model_input": False,
        "additional_augmentation_during_final_fit": False,
        "best_params": params,
        "final_epochs": final_epochs,
        "device": str(device),
        "test_metrics": test_metrics,
        "elapsed_seconds": round(time.time() - started, 3),
    })
    return {
        "Protocol": protocol,
        "Side": side,
        "Model": model_name,
        "Seed": seed,
        "TestAccuracy": test_metrics["accuracy"],
        "TestBalancedAccuracy": test_metrics["balanced_accuracy"],
        "TestSensitivity": test_metrics["sensitivity"],
        "TestSpecificity": test_metrics["specificity"],
        "TestF1Macro": test_metrics["f1_macro"],
        "TestROCAUC": test_metrics["roc_auc"],
        "TestPRAUC": test_metrics["pr_auc"],
        "RunDir": str(output_dir.resolve()),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", choices=[*PROTOCOLS, "all"], default="all")
    parser.add_argument("--side", choices=[*SIDES, "all"], default="all")
    parser.add_argument("--model", choices=[*MODELS, "all"], default="all")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--n-trials", type=int, default=10)
    parser.add_argument("--tune-epochs", type=int, default=8)
    parser.add_argument("--tune-patience", type=int, default=3)
    parser.add_argument("--final-epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-seeds", default="42,123,2026")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    protocols = PROTOCOLS if args.protocol == "all" else (args.protocol,)
    sides = SIDES if args.side == "all" else (args.side,)
    models = MODELS if args.model == "all" else (args.model,)
    eval_seeds = tuple(int(x.strip()) for x in str(args.eval_seeds).split(",") if x.strip())
    if args.folds < 2 or args.n_trials < 1:
        raise SystemExit("folds must be >=2 and n-trials must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_root.mkdir(parents=True, exist_ok=True)
    tuning_rows = []
    final_rows = []
    failures = []
    for protocol in protocols:
        for side in sides:
            train, test, train_path, test_path = source_pair(data_root, side)
            x, y, groups, _ = matrix(train)
            splits = grouped_splits(y, groups, requested=args.folds, seed=args.seed)
            for model_name in models:
                study_id = f"{protocol}_{side}_{model_name}"
                print(f"[{study_id}] trials={args.n_trials} folds={len(splits)}", flush=True)
                try:
                    sampler = optuna.samplers.TPESampler(seed=args.seed)
                    pruner = optuna.pruners.MedianPruner(n_startup_trials=2, n_warmup_steps=2)
                    study = optuna.create_study(
                        study_name=study_id, direction="maximize", sampler=sampler, pruner=pruner,
                    )
                    study.optimize(
                        lambda trial: objective_factory(
                            trial=trial, model_name=model_name, protocol=protocol, side=side,
                            train=train, splits=splits, device=device, seed=args.seed,
                            tune_epochs=args.tune_epochs, tune_patience=args.tune_patience,
                        ),
                        n_trials=args.n_trials,
                        catch=(RuntimeError, ValueError),
                    )
                    best = study.best_trial
                    tuning_row = {
                        "Protocol": protocol,
                        "Side": side,
                        "Model": model_name,
                        "Folds": len(splits),
                        "TrialsRequested": args.n_trials,
                        "TrialsCompleted": len(study.trials),
                        "BestOOFBalancedAccuracy": best.value,
                        "BestParams": json.dumps(json_ready(best.params), sort_keys=True),
                        "Study": study_id,
                    }
                    tuning_rows.append(tuning_row)
                    study_dir = output_root / "studies" / protocol / side / model_name
                    study_dir.mkdir(parents=True, exist_ok=True)
                    study.trials_dataframe().to_csv(study_dir / "trials.csv", index=False)
                    write_json(study_dir / "best_params.json", {
                        "protocol": protocol, "side": side, "model": model_name,
                        "folds": len(splits), "best_value_oof_balanced_accuracy": best.value,
                        "best_params": best.params,
                    })
                    print(f"  best OOF balanced_accuracy={best.value:.4f} params={best.params}", flush=True)
                    for final_seed in eval_seeds:
                        run_dir = output_root / "final" / protocol / side / model_name / f"seed_{final_seed}"
                        final_rows.append(train_final(
                            protocol=protocol, side=side, model_name=model_name,
                            params=best.params, train=train, test=test, device=device,
                            seed=final_seed, final_epochs=args.final_epochs,
                            output_dir=run_dir, train_path=train_path, test_path=test_path,
                        ))
                except Exception as exc:
                    print(f"  [FAIL] {study_id}: {exc!r}", flush=True)
                    failures.append({"Protocol": protocol, "Side": side, "Model": model_name, "Status": "FAIL", "Error": repr(exc)})
    tuning_path = output_root / "OPTUNA_TUNING_SUMMARY.csv"
    final_path = output_root / "OPTUNA_FINAL_SUMMARY.csv"
    pd.DataFrame(tuning_rows + failures).to_csv(tuning_path, index=False)
    pd.DataFrame(final_rows + failures).to_csv(final_path, index=False)
    winners = []
    if tuning_rows:
        tuning_frame = pd.DataFrame(tuning_rows)
        for (_, _), group in tuning_frame.groupby(["Protocol", "Side"]):
            winners.append(group.sort_values(["BestOOFBalancedAccuracy", "Model"], ascending=[False, True]).iloc[0].to_dict())
    pd.DataFrame(winners).to_csv(output_root / "OPTUNA_WINNERS_BY_PROTOCOL_SIDE.csv", index=False)
    write_json(output_root / "RUN_MANIFEST.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": data_root,
        "output_root": output_root,
        "protocols": protocols,
        "sides": sides,
        "models": models,
        "folds_requested": args.folds,
        "trials_per_study": args.n_trials,
        "tune_epochs": args.tune_epochs,
        "tune_patience": args.tune_patience,
        "final_epochs": args.final_epochs,
        "seed_for_folds_and_sampler": args.seed,
        "eval_seeds": eval_seeds,
        "device": str(device),
        "augmentation_inside_each_cv_fold": True,
        "test_used_during_tuning": False,
        "validation_rows_in_final_fit": 0,
        "pls_da_used": False,
        "failures": failures,
        "tuning_summary": tuning_path,
        "final_summary": final_path,
    })
    print(f"Wrote {tuning_path}")
    print(f"Wrote {final_path}")
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

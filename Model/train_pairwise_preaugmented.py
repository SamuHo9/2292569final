"""Train the six coefficient models on the pre-augmented fixed 80/20 sets.

This runner is deliberately different from the grouped-CV runners.  The
input train CSV already contains the requested pairwise synthetic rows, so it
does not augment again and it does not create a validation split.  The test
CSV is read once for final reporting only.  Each protocol/side/model/seed is
written to its own directory with a model artifact, train-loss log, test
predictions, metrics and a manifest.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

from leakage_free_coef_training import (
    MODELS,
    make_torch_model,
    metrics,
    seed_everything,
    tensor_for_model,
)


META = ["Subject", "Group", "Class", "BinaryClass", "DataType"]
FEATURES = [f"Coef_{i}" for i in range(1, 508)]
PROTOCOLS = ("between_class", "within_class")
SIDES = ("left", "right")


def write_json(path: Path, payload: dict) -> None:
    def convert(value):
        if isinstance(value, (np.integer,)):
            return int(value)
        if isinstance(value, (np.floating,)):
            return float(value)
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        return value

    path.write_text(json.dumps(convert(payload), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_frame(path: Path, split: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected = META + FEATURES
    if frame.columns.tolist() != expected:
        raise ValueError(f"Unexpected schema in {path}; expected {len(expected)} columns")
    if frame["Subject"].isna().any() or frame["Subject"].astype(str).duplicated().any():
        raise ValueError(f"Subject missing or duplicated in {path}")
    labels = pd.to_numeric(frame["BinaryClass"], errors="coerce")
    classes = pd.to_numeric(frame["Class"], errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        raise ValueError(f"Invalid BinaryClass in {path}")
    if not np.array_equal(labels.to_numpy(dtype=int), classes.to_numpy(dtype=int)):
        raise ValueError(f"Class/BinaryClass mismatch in {path}")
    if split == "test":
        if not frame["DataType"].astype(str).str.lower().isin(["original", "real"]).all():
            raise ValueError(f"Test contains synthetic rows: {path}")
    if not np.isfinite(frame[FEATURES].to_numpy(dtype=np.float64)).all():
        raise ValueError(f"Non-finite coefficient in {path}")
    result = frame.copy()
    result["BinaryClass"] = labels.astype(np.int64)
    result["Class"] = classes.astype(np.int64)
    return result


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = np.clip(values, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-values))


def train_neural(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    device: torch.device,
    seed: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    dropout: float,
    recon_weight: float,
) -> tuple[torch.nn.Module, np.ndarray, list[dict]]:
    seed_everything(seed)
    model = make_torch_model(name, dropout=dropout).to(device)
    counts = np.bincount(y_train, minlength=2).astype(np.float32)
    pos_weight = torch.tensor(
        [max(1.0, float(counts[0])) / max(1.0, float(counts[1]))],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    tx = tensor_for_model(name, x_train)
    ty = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    dataset = TensorDataset(tx, ty)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True, drop_last=False)
    log_rows: list[dict] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        total_rows = 0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            if name == "ResNetAE":
                logits, reconstruction = model(bx)
                loss = criterion(logits, by) + recon_weight * F.mse_loss(reconstruction, bx.flatten(1))
            else:
                logits = model(bx)
                loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(by)
            total_rows += len(by)
        log_rows.append({"epoch": epoch, "train_loss": total_loss / max(1, total_rows)})

    model.eval()
    with torch.no_grad():
        logits = model(tensor_for_model(name, x_test).to(device))
        if name == "ResNetAE":
            logits = logits[0]
        test_prob = torch.sigmoid(logits).flatten().cpu().numpy()
    return model, test_prob, log_rows


def evaluate(y_true: np.ndarray, probability: np.ndarray) -> dict:
    result = metrics(y_true, probability, threshold=0.5)
    predicted = (probability >= 0.5).astype(int)
    result["confusion_matrix"] = [
        [int(((y_true == 0) & (predicted == 0)).sum()), int(((y_true == 0) & (predicted == 1)).sum())],
        [int(((y_true == 1) & (predicted == 0)).sum()), int(((y_true == 1) & (predicted == 1)).sum())],
    ]
    return result


def run_one(
    *,
    protocol: str,
    side: str,
    model_name: str,
    seed: int,
    data_root: Path,
    output_root: Path,
    epochs: int,
    device: torch.device,
    learning_rate: float,
    weight_decay: float,
    batch_size: int,
    dropout: float,
    recon_weight: float,
) -> dict:
    started = time.time()
    side_cap = side.capitalize()
    train_path = data_root / protocol / side / "train" / f"All_coef_{side_cap}_train_coef_features.csv"
    test_path = data_root / protocol / side / "test" / f"All_coef_{side_cap}_test_coef_features.csv"
    train = load_frame(train_path, "train")
    test = load_frame(test_path, "test")
    train_subjects = set(train["Subject"].astype(str))
    test_subjects = set(test["Subject"].astype(str))
    if train_subjects & test_subjects:
        raise ValueError(f"Train/test Subject overlap: {protocol}/{side}")
    x_train = train[FEATURES].to_numpy(dtype=np.float32)
    y_train = train["BinaryClass"].to_numpy(dtype=np.int64)
    x_test = test[FEATURES].to_numpy(dtype=np.float32)
    y_test = test["BinaryClass"].to_numpy(dtype=np.int64)
    if not np.array_equal(np.sort(np.unique(y_train)), np.array([0, 1])):
        raise ValueError(f"Training set does not contain both classes: {protocol}/{side}")

    run_dir = output_root / protocol / side / model_name / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if model_name == "SVM":
        # probability=False avoids SVC's internal 5-fold calibration.  The
        # sigmoid below maps the decision function to a score for ROC/PR
        # metrics while the 0.5 threshold remains equivalent to decision=0.
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(C=1.0, kernel="rbf", gamma="scale", class_weight="balanced",
                           probability=False, random_state=seed)),
        ])
        model.fit(x_train, y_train)
        probability = sigmoid(model.decision_function(x_test))
        with (run_dir / "model.pkl").open("wb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pd.DataFrame([{"epoch": 1, "train_loss": None, "status": "fit_complete"}]).to_csv(
            run_dir / "training_log.csv", index=False
        )
        model_settings = {"C": 1.0, "gamma": "scale", "probability_calibration": "none", "score_mapping": "sigmoid(decision_function)"}
    else:
        scaler = StandardScaler().fit(x_train)
        scaled_train = scaler.transform(x_train).astype(np.float32)
        scaled_test = scaler.transform(x_test).astype(np.float32)
        model, probability, log_rows = train_neural(
            model_name, scaled_train, y_train, scaled_test, device, seed, epochs,
            learning_rate, weight_decay, batch_size, dropout, recon_weight,
        )
        torch.save(model.state_dict(), run_dir / "model_state.pt")
        with (run_dir / "scaler.pkl").open("wb") as handle:
            pickle.dump(scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)
        pd.DataFrame(log_rows).to_csv(run_dir / "training_log.csv", index=False)
        model_settings = {
            "optimizer": "AdamW",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
            "dropout": dropout,
            "reconstruction_weight": recon_weight if model_name == "ResNetAE" else None,
            "epochs": epochs,
            "scaler_fit_scope": "augmented training rows only",
        }
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    test_metrics = evaluate(y_test, probability)
    predictions = pd.DataFrame({
        "Subject": test["Subject"].astype(str),
        "BinaryClass": y_test,
        "Probability": probability,
        "Prediction": (probability >= 0.5).astype(int),
    })
    predictions.to_csv(run_dir / "test_predictions.csv", index=False)
    metrics_payload = {
        "test": test_metrics,
        "train_class_counts": {str(k): int(v) for k, v in train["BinaryClass"].value_counts().sort_index().items()},
        "test_class_counts": {str(k): int(v) for k, v in test["BinaryClass"].value_counts().sort_index().items()},
    }
    write_json(run_dir / "metrics.json", metrics_payload)
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "pairwise-balanced-train-only-v1",
        "augmentation_protocol": protocol,
        "dataset": "All_coef",
        "side": side,
        "model": model_name,
        "seed": seed,
        "train_csv": train_path.resolve(),
        "test_csv": test_path.resolve(),
        "train_rows": len(train),
        "test_rows": len(test),
        "features": len(FEATURES),
        "validation_created": False,
        "validation_rows": 0,
        "additional_augmentation_during_training": False,
        "pls_da_applied_to_model_input": False,
        "train_class_counts": metrics_payload["train_class_counts"],
        "test_class_counts": metrics_payload["test_class_counts"],
        "device": str(device),
        "model_settings": model_settings,
        "elapsed_seconds": round(time.time() - started, 3),
        "test_metrics_file": (run_dir / "metrics.json").resolve(),
    }
    write_json(run_dir / "run_manifest.json", manifest)
    return {
        "Protocol": protocol,
        "Side": side,
        "Model": model_name,
        "Seed": seed,
        "TrainRows": len(train),
        "TestRows": len(test),
        "TestAccuracy": test_metrics["accuracy"],
        "TestBalancedAccuracy": test_metrics["balanced_accuracy"],
        "TestSensitivity": test_metrics["sensitivity"],
        "TestSpecificity": test_metrics["specificity"],
        "TestF1Macro": test_metrics["f1_macro"],
        "TestROCAUC": test_metrics["roc_auc"],
        "TestPRAUC": test_metrics["pr_auc"],
        "RunDir": str(run_dir.resolve()),
        "Status": "OK",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", choices=[*PROTOCOLS, "all"], default="all")
    parser.add_argument("--side", choices=[*SIDES, "all"], default="all")
    parser.add_argument("--model", choices=[*MODELS, "all"], default="all")
    parser.add_argument("--seeds", default="42,123,2026")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--recon-weight", type=float, default=0.05)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    protocols = PROTOCOLS if args.protocol == "all" else (args.protocol,)
    sides = SIDES if args.side == "all" else (args.side,)
    models = MODELS if args.model == "all" else (args.model,)
    seeds = tuple(int(value.strip()) for value in str(args.seeds).split(",") if value.strip())
    if args.epochs < 1 or args.batch_size < 1:
        raise SystemExit("epochs and batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    output_root.mkdir(parents=True, exist_ok=True)
    rows = []
    failures = []
    total = len(protocols) * len(sides) * len(models) * len(seeds)
    current = 0
    for protocol in protocols:
        for side in sides:
            for model_name in models:
                for seed in seeds:
                    current += 1
                    print(f"[{current}/{total}] {protocol}/{side}/{model_name}/seed_{seed}", flush=True)
                    try:
                        row = run_one(
                            protocol=protocol, side=side, model_name=model_name, seed=seed,
                            data_root=data_root, output_root=output_root, epochs=args.epochs,
                            device=device, learning_rate=args.learning_rate,
                            weight_decay=args.weight_decay, batch_size=args.batch_size,
                            dropout=args.dropout, recon_weight=args.recon_weight,
                        )
                        rows.append(row)
                        print(
                            f"  test accuracy={row['TestAccuracy']:.4f} "
                            f"balanced_accuracy={row['TestBalancedAccuracy']:.4f}",
                            flush=True,
                        )
                    except Exception as exc:
                        print(f"  [FAIL] {exc!r}", flush=True)
                        failure = {
                            "Protocol": protocol, "Side": side, "Model": model_name,
                            "Seed": seed, "Status": "FAIL", "Error": repr(exc),
                        }
                        rows.append(failure)
                        failures.append(failure)
    summary_path = output_root / "PAIRWISE_TRAIN_SUMMARY.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    run_manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": data_root,
        "output_root": output_root,
        "protocols": protocols,
        "sides": sides,
        "models": models,
        "seeds": seeds,
        "epochs": args.epochs,
        "device": str(device),
        "validation_created": False,
        "additional_augmentation_during_training": False,
        "pls_da_used": False,
        "optimizer_neural": "AdamW",
        "summary": summary_path,
        "failures": failures,
        "run_count": len(rows),
    }
    write_json(output_root / "RUN_MANIFEST.json", run_manifest)
    print(f"Wrote summary: {summary_path}")
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

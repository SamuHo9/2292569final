"""Architecture-aware, leakage-free Optuna runner for coefficient models.

This runner is deliberately separate from ``optuna_leakage_free_all.py``.  The
existing runner is the fixed-topology baseline.  This file searches a small,
predeclared architecture space while keeping the data protocol identical:
patient-grouped folds, fold-local scaling/PLS-DA/augmentation, and an untouched
test set.  The test set is never read by Optuna and the final threshold is
selected from the best trial's training OOF predictions only.

The supported protocols are ``coef_raw`` and ``coef_plsda``.  PointNet is kept
in its own runner because it consumes XYZ point clouds rather than 507
coefficient features.
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

from data_contract import load_raw_cohort, patient_group_id
from leakage_free_coef_training import (
    COHORTS,
    MODELS,
    SIDES,
    augment_training_fold,
    grouped_splits,
    matrix,
    metrics,
    seed_everything,
)
from leakage_free_plsda_training import plsda_augment_training_fold


PROTOCOLS = ("coef_raw", "coef_plsda")


def jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    return value


def activation(name: str) -> nn.Module:
    return nn.GELU() if name == "gelu" else nn.ReLU()


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, activation_name: str):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.norm1 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding, bias=False)
        self.norm2 = nn.GroupNorm(1, channels)
        self.act = activation(activation_name)

    def forward(self, x):
        out = self.act(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.act(out + x)


class DynamicResNet1D(nn.Module):
    def __init__(self, params, autoencoder: bool = False):
        super().__init__()
        base = int(params["base_channels"])
        kernel = int(params["kernel_size"])
        act_name = params["activation"]
        blocks = int(params["residual_blocks"])
        pool_size = int(params["pool_size"])
        dropout = float(params["dropout"])
        channels = [base, base * 2, base * 4]

        stem_padding = kernel // 2
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels[0], kernel, padding=stem_padding),
            nn.GroupNorm(1, channels[0]),
            activation(act_name),
        )
        body = []
        for stage, out_channels in enumerate(channels):
            if stage > 0:
                body.extend([
                    nn.Conv1d(channels[stage - 1], out_channels, kernel,
                              stride=2, padding=stem_padding, bias=False),
                    nn.GroupNorm(1, out_channels),
                    activation(act_name),
                ])
            body.extend(ResidualBlock(out_channels, kernel, act_name) for _ in range(blocks))
        self.body = nn.Sequential(*body)
        self.pool = nn.AdaptiveAvgPool1d(pool_size)
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout), nn.Linear(channels[-1] * pool_size, 1)
        )
        self.autoencoder = bool(autoencoder)
        if self.autoencoder:
            self.decode = nn.Sequential(
                nn.Linear(channels[-1], int(params["bottleneck"])),
                activation(act_name),
                nn.Linear(int(params["bottleneck"]), 507),
            )

    def forward(self, x):
        latent = self.body(self.stem(x))
        logits = self.head(self.pool(latent))
        if self.autoencoder:
            pooled = F.adaptive_avg_pool1d(latent, 1).flatten(1)
            return logits, self.decode(pooled)
        return logits


class DepthwiseSeparable(nn.Module):
    def __init__(self, inp, out, kernel, stride, act_name):
        super().__init__()
        padding = kernel // 2
        self.block = nn.Sequential(
            nn.Conv1d(inp, inp, kernel, stride=stride, padding=padding,
                      groups=inp, bias=False),
            nn.GroupNorm(1, inp),
            activation(act_name),
            nn.Conv1d(inp, out, 1, bias=False),
            nn.GroupNorm(1, out),
            activation(act_name),
        )

    def forward(self, x):
        return self.block(x)


class DynamicMobileNet1D(nn.Module):
    def __init__(self, params):
        super().__init__()
        mult = float(params["width_multiplier"])
        kernel = int(params["kernel_size"])
        depth = int(params["depth_multiplier"])
        act_name = params["activation"]
        base = [max(4, int(round(v * mult))) for v in (16, 32, 64, 128)]
        layers = [nn.Conv1d(1, base[0], 3, padding=1, bias=False),
                  nn.GroupNorm(1, base[0]), activation(act_name)]
        inp = base[0]
        for stage, out in enumerate(base):
            for repeat in range(depth):
                stride = 2 if repeat == 0 and stage > 0 else 1
                layers.append(DepthwiseSeparable(inp, out, kernel, stride, act_name))
                inp = out
        self.features = nn.Sequential(*layers)
        pool_size = int(params["pool_size"])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(pool_size), nn.Flatten(),
            nn.Dropout(float(params["dropout"])), nn.Linear(inp * pool_size, 1)
        )

    def forward(self, x):
        return self.head(self.features(x))


class FireBlock(nn.Module):
    def __init__(self, inp, squeeze, out, kernel_size):
        super().__init__()
        self.squeeze = nn.Conv1d(inp, squeeze, 1)
        self.e1 = nn.Conv1d(squeeze, out // 2, 1)
        self.e3 = nn.Conv1d(squeeze, out // 2, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        x = F.relu(self.squeeze(x))
        return torch.cat([F.relu(self.e1(x)), F.relu(self.e3(x))], dim=1)


class DynamicSqueezeNet1D(nn.Module):
    def __init__(self, params):
        super().__init__()
        width = int(params["squeeze_width"])
        fire_blocks = int(params["fire_blocks"])
        kernel = int(params["kernel_size"])
        layers = [nn.Conv1d(1, width, 3, padding=1), nn.ReLU()]
        inp = width
        for idx in range(fire_blocks):
            out = width * (2 ** min(idx + 1, 3))
            squeeze = max(2, out // int(params["squeeze_ratio"]))
            layers.append(FireBlock(inp, squeeze, out, kernel))
            inp = out
        self.features = nn.Sequential(*layers)
        pool_size = int(params["pool_size"])
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(pool_size), nn.Flatten(),
            nn.Dropout(float(params["dropout"])), nn.Linear(inp * pool_size, 1)
        )

    def forward(self, x):
        return self.head(self.features(x))


class DynamicMLP(nn.Module):
    def __init__(self, params):
        super().__init__()
        widths = [int(v) for v in params["hidden_widths"]]
        layers = []
        inp = 507
        for width in widths:
            layers.extend([nn.Linear(inp, width), nn.LayerNorm(width),
                           activation(params["activation"]),
                           nn.Dropout(float(params["dropout"]))])
            inp = width
        layers.append(nn.Linear(inp, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(1))


def make_model(name, params):
    if name == "MLP":
        return DynamicMLP(params)
    if name == "ResNet":
        return DynamicResNet1D(params)
    if name == "ResNetAE":
        return DynamicResNet1D(params, autoencoder=True)
    if name == "MobileNet":
        return DynamicMobileNet1D(params)
    if name == "SqueezeNet":
        return DynamicSqueezeNet1D(params)
    raise ValueError(f"Unsupported neural model: {name}")


def tensor_for_model(name, values):
    tensor = torch.tensor(values, dtype=torch.float32)
    return tensor if name == "MLP" else tensor.unsqueeze(1)


def choose_params(trial, model_name):
    params = {
        "lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
        "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "dropout": trial.suggest_float("dropout", 0.10, 0.50),
        "activation": trial.suggest_categorical("activation", ["relu", "gelu"]),
        "pls_components": trial.suggest_categorical(
            "pls_components", [2, 4, 6, 8, 12, 16, 24]),
    }
    if model_name == "MLP":
        depth = trial.suggest_int("mlp_depth", 1, 3)
        width = trial.suggest_categorical("mlp_width", [32, 64, 128, 256])
        params["hidden_widths"] = tuple(int(width) for _ in range(depth))
    elif model_name in ("ResNet", "ResNetAE"):
        params.update({
            "base_channels": trial.suggest_categorical("base_channels", [16, 32, 64]),
            "kernel_size": trial.suggest_categorical("kernel_size", [3, 5, 7]),
            "residual_blocks": trial.suggest_int("residual_blocks", 1, 3),
            "pool_size": trial.suggest_categorical("pool_size", [1, 4, 8]),
        })
        if model_name == "ResNetAE":
            params.update({
                "bottleneck": trial.suggest_categorical("bottleneck", [16, 32, 64, 128]),
                "recon_weight": trial.suggest_float("recon_weight", 0.0, 0.20),
            })
    elif model_name == "MobileNet":
        params.update({
            "width_multiplier": trial.suggest_categorical("width_multiplier", [0.5, 1.0, 1.5]),
            "depth_multiplier": trial.suggest_int("depth_multiplier", 1, 3),
            "kernel_size": trial.suggest_categorical("kernel_size", [3, 5]),
            "pool_size": trial.suggest_categorical("pool_size", [1, 4, 8]),
        })
    elif model_name == "SqueezeNet":
        params.update({
            "squeeze_width": trial.suggest_categorical("squeeze_width", [8, 16, 32]),
            "fire_blocks": trial.suggest_int("fire_blocks", 2, 4),
            "squeeze_ratio": trial.suggest_categorical("squeeze_ratio", [2, 4, 8]),
            "kernel_size": trial.suggest_categorical("kernel_size", [3, 5]),
            "pool_size": trial.suggest_categorical("pool_size", [1, 4, 8]),
        })
    elif model_name == "SVM":
        params = {
            "kernel": trial.suggest_categorical("kernel", ["linear", "rbf"]),
            "C": trial.suggest_float("C", 1e-3, 1e3, log=True),
            "gamma_mode": trial.suggest_categorical("gamma_mode", ["scale", "auto", "numeric"]),
            "pls_components": trial.suggest_categorical(
                "pls_components", [2, 4, 6, 8, 12, 16, 24]),
        }
        if params["gamma_mode"] == "numeric":
            params["gamma"] = trial.suggest_float("gamma", 1e-6, 1e-1, log=True)
    else:
        raise ValueError(model_name)
    return params


def materialize_params(raw_params, model_name):
    """Rebuild derived architecture values from Optuna's flat trial params."""
    params = dict(raw_params)
    if model_name == "MLP":
        depth = int(params["mlp_depth"])
        width = int(params["mlp_width"])
        params["hidden_widths"] = tuple(width for _ in range(depth))
    return params


def svm_model(params, seed):
    kernel = params.get("kernel", "rbf")
    gamma = params.get("gamma_mode", "scale")
    if gamma == "numeric":
        gamma = float(params["gamma"])
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(C=float(params["C"]), kernel=kernel, gamma=gamma,
                       class_weight="balanced", probability=True, random_state=seed)),
    ])


def augmented_fold(x, y, seed, protocol, params, children_per_pair, noise_scale):
    if protocol == "coef_plsda":
        return plsda_augment_training_fold(
            x, y, seed, children_per_pair, int(params["pls_components"])
        )
    return augment_training_fold(
        x, y, seed, "balanced_jitter", noise_scale, children_per_pair
    )


def train_neural_fold(name, params, x_train, y_train, x_val, y_val,
                      device, seed, epochs, patience):
    seed_everything(seed)
    model = make_model(name, params).to(device)
    counts = np.bincount(y_train, minlength=2).astype(np.float32)
    pos_weight = torch.tensor(
        [max(1.0, counts[0]) / max(1.0, counts[1])],
        dtype=torch.float32, device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )
    tx = tensor_for_model(name, x_train)
    ty = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    vx = tensor_for_model(name, x_val)
    vy = torch.tensor(y_val, dtype=torch.float32).view(-1, 1)
    loader = DataLoader(
        TensorDataset(tx, ty), batch_size=min(int(params["batch_size"]), len(ty)),
        shuffle=True, drop_last=False,
    )
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    for epoch in range(1, int(epochs) + 1):
        model.train()
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(bx)
            logits = output[0] if name == "ResNetAE" else output
            loss = criterion(logits, by)
            if name == "ResNetAE":
                loss = loss + float(params.get("recon_weight", 0.05)) * F.mse_loss(
                    output[1], bx.flatten(1)
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            output = model(vx.to(device))
            logits = output[0] if name == "ResNetAE" else output
            val_loss = criterion(logits, vy.to(device)).item()
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        output = model(vx.to(device))
        logits = output[0] if name == "ResNetAE" else output
        probability = torch.sigmoid(logits).flatten().cpu().numpy()
    return model, probability, best_epoch


def train_neural_full(name, params, x_train, y_train, device, seed, epochs):
    seed_everything(seed)
    model = make_model(name, params).to(device)
    counts = np.bincount(y_train, minlength=2).astype(np.float32)
    pos_weight = torch.tensor(
        [max(1.0, counts[0]) / max(1.0, counts[1])],
        dtype=torch.float32, device=device,
    )
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(params["lr"]),
        weight_decay=float(params["weight_decay"]),
    )
    tx = tensor_for_model(name, x_train)
    ty = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    batch = min(int(params["batch_size"]), len(ty))
    loader = DataLoader(TensorDataset(tx, ty), batch_size=batch,
                        shuffle=True, drop_last=False)
    model.train()
    for _ in range(max(1, int(epochs))):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(bx)
            logits = output[0] if name == "ResNetAE" else output
            loss = criterion(logits, by)
            if name == "ResNetAE":
                loss = loss + float(params.get("recon_weight", 0.05)) * F.mse_loss(
                    output[1], bx.flatten(1)
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    model.eval()
    return model


def oof_predictions(model_name, protocol, x, y, splits, params, args,
                    seed_offset=0):
    oof = np.zeros(len(y), dtype=float)
    fold_info = []
    for fold, (tr, va) in enumerate(splits):
        fold_seed = int(args.seed + seed_offset + fold)
        fit_x, fit_y, aug_info = augmented_fold(
            x[tr], y[tr], fold_seed, protocol, params,
            args.children_per_pair, args.noise_scale,
        )
        if model_name == "SVM":
            model = svm_model(params, fold_seed).fit(fit_x, fit_y)
            oof[va] = model.predict_proba(x[va])[:, 1]
        else:
            scaler = StandardScaler().fit(fit_x)
            xtr = scaler.transform(fit_x).astype(np.float32)
            xva = scaler.transform(x[va]).astype(np.float32)
            model, val_prob, best_epoch = train_neural_fold(
                model_name, params, xtr, fit_y, xva, y[va], args.device_obj,
                fold_seed, args.tune_epochs, args.tune_patience,
            )
            oof[va] = val_prob
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            aug_info["best_epoch"] = int(best_epoch)
        aug_info.update({"fold": int(fold + 1), "raw_train_n": int(len(tr)),
                         "val_n": int(len(va))})
        fold_info.append(aug_info)
    return oof, fold_info


def choose_threshold(y, probability):
    candidates = np.linspace(0.10, 0.90, 161)
    scored = []
    for threshold in candidates:
        pred = (probability >= threshold).astype(int)
        ba = metrics(y, probability, float(threshold))["balanced_accuracy"]
        f1 = f1_score(y, pred, average="macro", zero_division=0)
        scored.append((float(ba), float(f1), -abs(float(threshold) - 0.5), float(threshold)))
    return max(scored)[-1]


def final_test(model_name, protocol, params, train_df, test_df, x, y, test_x,
               args, seed, threshold, run_dir):
    started = time.time()
    fit_x, fit_y, aug_info = augmented_fold(
        x, y, seed, protocol, params, args.children_per_pair, args.noise_scale
    )
    if model_name == "SVM":
        model = svm_model(params, seed).fit(fit_x, fit_y)
        probability = model.predict_proba(test_x)[:, 1]
        with (run_dir / "model.pkl").open("wb") as stream:
            pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        scaler = StandardScaler().fit(fit_x)
        scaled_x = scaler.transform(fit_x).astype(np.float32)
        scaled_test = scaler.transform(test_x).astype(np.float32)
        model = train_neural_full(
            model_name, params, scaled_x, fit_y, args.device_obj,
            seed, args.final_epochs,
        )
        with torch.no_grad():
            output = model(tensor_for_model(model_name, scaled_test).to(args.device_obj))
            logits = output[0] if model_name == "ResNetAE" else output
            probability = torch.sigmoid(logits).flatten().cpu().numpy()
        torch.save(model.state_dict(), run_dir / "model_state.pt")
        with (run_dir / "scaler.pkl").open("wb") as stream:
            pickle.dump(scaler, stream, protocol=pickle.HIGHEST_PROTOCOL)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    test_y = test_df["BinaryClass"].to_numpy(dtype=int)
    default_metrics = metrics(test_y, probability, 0.5)
    tuned_metrics = metrics(test_y, probability, threshold)
    pred = pd.DataFrame({
        "Subject": test_df["Subject"].astype(str),
        "PatientID": [patient_group_id(v) for v in test_df["Subject"]],
        "BinaryClass": test_y,
        "Probability": probability,
        "Prediction_0_5": (probability >= 0.5).astype(int),
        "Prediction": (probability >= threshold).astype(int),
    })
    pred.to_csv(run_dir / "test_predictions.csv", index=False)
    (run_dir / "metrics.json").write_text(json.dumps({
        "test_default_0_5": default_metrics,
        "test_tuned_oof_threshold": tuned_metrics,
    }, indent=2), encoding="utf-8")
    (run_dir / "run_manifest.json").write_text(json.dumps({
        "protocol": protocol,
        "cohort": str(train_df["Cohort"].iloc[0]) if "Cohort" in train_df else "",
        "side": args.current_side,
        "model": model_name,
        "architecture_search": True,
        "architecture_params": jsonable(params),
        "threshold_source": "training_oof_only",
        "threshold": float(threshold),
        "folds": int(args.folds),
        "seed": int(seed),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "data_root": str(args.data_root),
        "test_used_during_tuning": False,
        "augmentation_applied_to_test": False,
        "augmentation_applied_to_validation": False,
        "pls_da_fit_scope": "training_data_only" if protocol == "coef_plsda" else None,
        "augmentation": aug_info,
        "elapsed_seconds": round(time.time() - started, 2),
    }, indent=2), encoding="utf-8")
    return default_metrics, tuned_metrics


def study_one(protocol, model_name, cohort, side, args, output_root):
    train_df, test_df = load_raw_cohort(cohort, side, kind="coef", root=args.data_root)
    x, y, groups, columns = matrix(train_df)
    test_x, test_y, test_groups, _ = matrix(test_df)
    if set(groups) & set(test_groups):
        raise ValueError(f"Patient overlap: {cohort}/{side}")
    splits = grouped_splits(y, groups, args.folds, args.seed)
    study_dir = output_root / "studies" / protocol / cohort / side / model_name
    study_dir.mkdir(parents=True, exist_ok=True)
    study_name = f"architecture_{protocol}_{cohort}_{side}_{model_name}"
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3),
        study_name=study_name,
    )

    def objective(trial):
        params = choose_params(trial, model_name)
        oof = np.zeros(len(y), dtype=float)
        for fold, (tr, va) in enumerate(splits):
            fold_seed = int(args.seed + trial.number * 1000 + fold)
            fit_x, fit_y, _ = augmented_fold(
                x[tr], y[tr], fold_seed, protocol, params,
                args.children_per_pair, args.noise_scale,
            )
            if model_name == "SVM":
                model = svm_model(params, fold_seed).fit(fit_x, fit_y)
                oof[va] = model.predict_proba(x[va])[:, 1]
            else:
                scaler = StandardScaler().fit(fit_x)
                model, val_prob, _ = train_neural_fold(
                    model_name, params,
                    scaler.transform(fit_x).astype(np.float32), fit_y,
                    scaler.transform(x[va]).astype(np.float32), y[va],
                    args.device_obj, fold_seed, args.tune_epochs,
                    args.tune_patience,
                )
                oof[va] = val_prob
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            score = metrics(y, oof, 0.5)["balanced_accuracy"]
            trial.report(float(score), step=fold)
            if trial.should_prune():
                raise optuna.TrialPruned()
        return float(metrics(y, oof, 0.5)["balanced_accuracy"])

    study.optimize(objective, n_trials=args.n_trials, n_jobs=1,
                   catch=(ValueError, RuntimeError, FloatingPointError))
    if study.best_trial is None:
        raise RuntimeError(f"No completed trial: {study_name}")
    best_params = materialize_params(study.best_trial.params, model_name)
    best_params = jsonable(best_params)
    study.trials_dataframe().to_csv(study_dir / "trials.csv", index=False)
    (study_dir / "best_params.json").write_text(json.dumps({
        "protocol": protocol, "cohort": cohort, "side": side,
        "model": model_name, "folds": len(splits),
        "n_trials_requested": args.n_trials,
        "n_trials_completed": len([t for t in study.trials
                                    if t.state == optuna.trial.TrialState.COMPLETE]),
        "best_trial": int(study.best_trial.number),
        "best_oof_balanced_accuracy_0_5": float(study.best_value),
        "best_params": best_params,
    }, indent=2), encoding="utf-8")

    oof, fold_info = oof_predictions(
        model_name, protocol, x, y, splits, best_params, args, seed_offset=100000
    )
    threshold = choose_threshold(y, oof)
    oof_metrics_default = metrics(y, oof, 0.5)
    oof_metrics_tuned = metrics(y, oof, threshold)
    pd.DataFrame({
        "Subject": train_df["Subject"].astype(str),
        "PatientID": groups,
        "BinaryClass": y,
        "Probability": oof,
        "Prediction_0_5": (oof >= 0.5).astype(int),
        "Prediction": (oof >= threshold).astype(int),
    }).to_csv(study_dir / "best_oof_predictions.csv", index=False)
    (study_dir / "threshold.json").write_text(json.dumps({
        "threshold": float(threshold),
        "source": "best_trial_training_oof_only",
        "oof_default": oof_metrics_default,
        "oof_tuned": oof_metrics_tuned,
        "folds": fold_info,
    }, indent=2), encoding="utf-8")

    final_rows = []
    args.current_side = side
    for seed in args.eval_seeds:
        run_dir = output_root / "final" / protocol / cohort / side / model_name / f"seed_{seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        default_metrics, tuned_metrics = final_test(
            model_name, protocol, best_params, train_df, test_df, x, y,
            test_x, args, seed, threshold, run_dir,
        )
        final_rows.append({
            "Protocol": protocol, "Cohort": cohort, "Side": side,
            "Model": model_name, "EvalSeed": int(seed),
            "OptunaTrials": int(args.n_trials),
            "BestOOF_BalancedAccuracy_0_5": float(study.best_value),
            "OOF_BalancedAccuracy_Tuned": oof_metrics_tuned["balanced_accuracy"],
            "Threshold": float(threshold),
            "Test_Accuracy_0_5": default_metrics["accuracy"],
            "Test_BalancedAccuracy_0_5": default_metrics["balanced_accuracy"],
            "Test_F1_Macro_0_5": default_metrics["f1_macro"],
            "Test_Accuracy_Tuned": tuned_metrics["accuracy"],
            "Test_BalancedAccuracy_Tuned": tuned_metrics["balanced_accuracy"],
            "Test_Sensitivity_Tuned": tuned_metrics["sensitivity"],
            "Test_Specificity_Tuned": tuned_metrics["specificity"],
            "Test_F1_Macro_Tuned": tuned_metrics["f1_macro"],
            "Test_ROC_AUC": tuned_metrics["roc_auc"],
            "Test_PR_AUC": tuned_metrics["pr_auc"],
            "BestParams": json.dumps(best_params, sort_keys=True),
            "RunDir": str(run_dir),
        })
    return final_rows, {
        "Protocol": protocol, "Cohort": cohort, "Side": side,
        "Model": model_name, "OptunaTrials": int(args.n_trials),
        "BestTrial": int(study.best_trial.number),
        "BestOOF_BalancedAccuracy_0_5": float(study.best_value),
        "OOF_BalancedAccuracy_Tuned": oof_metrics_tuned["balanced_accuracy"],
        "Threshold": float(threshold),
        "BestParams": json.dumps(best_params, sort_keys=True),
        "StudyDir": str(study_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="Optuna architecture search for coefficient models")
    parser.add_argument("--protocol", choices=PROTOCOLS, default="coef_plsda")
    parser.add_argument("--cohort", choices=COHORTS + ("all",), default="all")
    parser.add_argument("--side", choices=SIDES + ("all",), default="all")
    parser.add_argument("--model", choices=MODELS + ("all",), default="all")
    parser.add_argument("--folds", type=int, default=10)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--tune_epochs", type=int, default=20)
    parser.add_argument("--tune_patience", type=int, default=5)
    parser.add_argument("--final_epochs", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval_seeds", default="42,123,2026")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--children_per_pair", type=int, default=8)
    parser.add_argument("--noise_scale", type=float, default=0.02)
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output_root", required=True)
    args = parser.parse_args()
    args.data_root = str(Path(args.data_root).resolve())
    args.output_root = str(Path(args.output_root).resolve())
    args.eval_seeds = tuple(int(v.strip()) for v in str(args.eval_seeds).split(",") if v.strip())
    args.device_obj = torch.device(
        args.device if args.device != "auto" else
        ("cuda" if torch.cuda.is_available() else "cpu")
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    cohorts = COHORTS if args.cohort == "all" else (args.cohort,)
    sides = SIDES if args.side == "all" else (args.side,)
    models = MODELS if args.model == "all" else (args.model,)
    all_final, all_tuning, failures = [], [], []
    for cohort in cohorts:
        for side in sides:
            for model_name in models:
                try:
                    print(f"[architecture/{args.protocol}/{cohort}/{side}/{model_name}] "
                          f"10-fold trials={args.n_trials}", flush=True)
                    final_rows, tuning_row = study_one(
                        args.protocol, model_name, cohort, side, args, output_root
                    )
                    all_final.extend(final_rows)
                    all_tuning.append(tuning_row)
                    print(f"  best OOF BA={tuning_row['BestOOF_BalancedAccuracy_0_5']:.4f} "
                          f"threshold={tuning_row['Threshold']:.3f}", flush=True)
                except Exception as exc:
                    print(f"[FAIL] {args.protocol}/{cohort}/{side}/{model_name}: {exc}", flush=True)
                    failures.append({
                        "Protocol": args.protocol, "Cohort": cohort, "Side": side,
                        "Model": model_name, "Status": "FAIL", "Error": repr(exc),
                    })
    tuning_path = output_root / "ARCH_OPT_TUNING_SUMMARY.csv"
    final_path = output_root / "ARCH_OPT_FINAL_SUMMARY.csv"
    pd.DataFrame(all_tuning + failures).to_csv(tuning_path, index=False)
    pd.DataFrame(all_final + failures).to_csv(final_path, index=False)
    winners = []
    if all_tuning:
        frame = pd.DataFrame(all_tuning)
        for _, group in frame.groupby(["Cohort", "Side"], sort=True):
            winners.append(group.sort_values(
                ["OOF_BalancedAccuracy_Tuned", "Model"],
                ascending=[False, True],
            ).iloc[0].to_dict())
    winner_path = output_root / "ARCH_OPT_WINNERS_BY_COHORT_SIDE.csv"
    pd.DataFrame(winners).to_csv(winner_path, index=False)
    (output_root / "RUN_MANIFEST.json").write_text(json.dumps({
        "protocol": args.protocol,
        "cohorts": cohorts,
        "sides": sides,
        "models": models,
        "data_root": args.data_root,
        "folds_requested": args.folds,
        "trials_per_study": args.n_trials,
        "test_used_during_tuning": False,
        "threshold_source": "training_oof_only",
        "eval_seeds": args.eval_seeds,
        "device": str(args.device_obj),
        "failures": failures,
    }, indent=2), encoding="utf-8")
    print(f"Wrote {tuning_path}")
    print(f"Wrote {final_path}")
    print(f"Wrote {winner_path}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Optuna tuning with grouped out-of-fold validation for every model family.

The script keeps the existing fixed-hyperparameter runners intact and writes a
separate result tree.  Every study is scoped to one cohort, one side, one
protocol and one model.  The objective is the balanced accuracy of grouped
10-fold out-of-fold predictions.  The held-out test set is never read during
an objective.  After the best trial is frozen, one final model is fit on the
complete training cohort and tested once per requested seed; fold models are
never averaged.

Supported protocols:
  coef_raw       six coefficient models with fold-local jitter augmentation
  coef_plsda     six coefficient models with fold-local PLS-DA augmentation
  pointnet_raw   raw XYZ PointNet with fold-local jitter augmentation
  pointnet_plsda PointNet with fold-local PLS-DA XYZ augmentation
  plsda_direct   direct PLS-DA classifier (n_components is tuned)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.base import clone
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from torch.utils.data import DataLoader, TensorDataset

from data_contract import (RAW_COEF_PROTOCOL, load_raw_cohort,
                           load_raw_xyz_cohort, patient_group_id)
from leakage_free_coef_training import (
    COHORTS, MODELS, SIDES, augment_training_fold, grouped_splits, make_torch_model,
    matrix, metrics, seed_everything, tensor_for_model, train_torch_fold,
)
from leakage_free_plsda_training import plsda_augment_training_fold
from leakage_free_pointnet_training import augment_xyz_fold, train_fold, xyz_matrix
from leakage_free_pointnet_plsda_training import plsda_augment_xyz_fold
from leakage_free_plsda_classifier import fit_plsda, predict_probability


COEF_MODELS = tuple(MODELS)
PROTOCOLS = ('coef_raw', 'coef_plsda', 'pointnet_raw', 'pointnet_plsda', 'plsda_direct')
PROTOCOL_LABELS = {
    'coef_raw': 'Optuna no-PLS coefficient',
    'coef_plsda': 'Optuna fold-local PLS-DA coefficient',
    'pointnet_raw': 'Optuna raw XYZ PointNet',
    'pointnet_plsda': 'Optuna fold-local PLS-DA PointNet',
    'plsda_direct': 'Optuna direct PLS-DA classifier',
}


def digest_frame(frame):
    return hashlib.sha256(frame.to_csv(index=False).encode('utf-8')).hexdigest()


def jsonable(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def choose_params(trial, protocol, model_name):
    """Search spaces shared across cohort/side studies."""
    if protocol == 'plsda_direct':
        return {'n_components': trial.suggest_categorical(
            'n_components', [2, 4, 6, 8, 12, 16, 24])}
    if protocol in ('coef_raw', 'coef_plsda') and model_name == 'SVM':
        gamma_mode = trial.suggest_categorical('gamma_mode', ['scale', 'auto', 'numeric'])
        params = {
            'C': trial.suggest_float('C', 1e-2, 1e2, log=True),
            'gamma_mode': gamma_mode,
        }
        if gamma_mode == 'numeric':
            params['gamma'] = trial.suggest_float('gamma', 1e-5, 1e-1, log=True)
        return params
    if protocol in ('coef_raw', 'coef_plsda'):
        params = {
            'lr': trial.suggest_float('lr', 1e-4, 3e-3, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32]),
            'dropout': trial.suggest_float('dropout', 0.10, 0.50),
        }
        if model_name == 'ResNetAE':
            params['recon_weight'] = trial.suggest_float('recon_weight', 0.01, 0.20, log=True)
        return params
    if protocol in ('pointnet_raw', 'pointnet_plsda'):
        return {
            'lr': trial.suggest_float('lr', 1e-4, 3e-3, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [8, 16, 32]),
            'dropout': trial.suggest_float('dropout', 0.10, 0.60),
        }
    raise ValueError(f'Unsupported protocol/model: {protocol}/{model_name}')


def coef_training_fold(x, y, seed, protocol, children_per_pair, noise_scale, pls_components):
    if protocol == 'coef_raw':
        return augment_training_fold(x, y, seed, 'balanced_jitter', noise_scale,
                                     children_per_pair)
    if protocol == 'coef_plsda':
        return plsda_augment_training_fold(x, y, seed, children_per_pair, pls_components)
    raise ValueError(protocol)


def svm_model(params, seed):
    gamma = params.get('gamma_mode', 'scale')
    if gamma == 'numeric':
        gamma = float(params['gamma'])
    return Pipeline([
        ('scaler', StandardScaler()),
        ('model', SVC(C=float(params['C']), kernel='rbf', gamma=gamma,
                      class_weight='balanced', probability=True, random_state=seed)),
    ])


def objective_coef(trial, model_name, protocol, x, y, splits, args):
    params = choose_params(trial, protocol, model_name)
    oof = np.zeros(len(y), dtype=float)
    for fold, (tr, va) in enumerate(splits):
        fold_seed = args.seed + trial.number * 1000 + fold
        if model_name == 'SVM':
            fit_x, fit_y, _ = coef_training_fold(
                x[tr], y[tr], fold_seed, protocol, args.children_per_pair,
                args.noise_scale, args.pls_components)
            model = svm_model(params, fold_seed).fit(fit_x, fit_y)
            oof[va] = model.predict_proba(x[va])[:, 1]
        else:
            fit_x, fit_y, _ = coef_training_fold(
                x[tr], y[tr], fold_seed, protocol, args.children_per_pair,
                args.noise_scale, args.pls_components)
            scaler = StandardScaler().fit(fit_x)
            xtr = scaler.transform(fit_x).astype(np.float32)
            xva = scaler.transform(x[va]).astype(np.float32)
            _, val_prob, _ = train_torch_fold(
                model_name, xtr, fit_y, xva, y[va], args.device_obj,
                fold_seed, args.tune_epochs, args.tune_patience,
                lr=params['lr'], weight_decay=params['weight_decay'],
                batch_size=params['batch_size'],
                recon_weight=params.get('recon_weight', 0.05),
                dropout=params.get('dropout'))
            oof[va] = val_prob
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        score = metrics(y, oof)['balanced_accuracy']
        trial.report(float(score), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(metrics(y, oof)['balanced_accuracy'])


def objective_pointnet(trial, protocol, x, y, splits, args):
    params = choose_params(trial, protocol, 'PointNet')
    oof = np.zeros(len(y), dtype=float)
    for fold, (tr, va) in enumerate(splits):
        fold_seed = args.seed + trial.number * 1000 + fold
        if protocol == 'pointnet_raw':
            fit_x, fit_y, _ = augment_xyz_fold(
                x[tr], y[tr], fold_seed, 'balanced_jitter', args.noise_scale,
                args.children_per_pair)
        else:
            fit_x, fit_y, _ = plsda_augment_xyz_fold(
                x[tr], y[tr], fold_seed, args.children_per_pair, args.pls_components)
        _, val_prob, _ = train_fold(
            fit_x, fit_y, x[va], y[va], args.device_obj, fold_seed,
            args.tune_epochs, args.tune_patience,
            lr=params['lr'], weight_decay=params['weight_decay'],
            batch_size=params['batch_size'], dropout=params['dropout'])
        oof[va] = val_prob
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        score = metrics(y, oof)['balanced_accuracy']
        trial.report(float(score), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(metrics(y, oof)['balanced_accuracy'])


def objective_direct(trial, x, y, splits, args):
    params = choose_params(trial, 'plsda_direct', 'PLSDA_classifier')
    oof = np.zeros(len(y), dtype=float)
    for fold, (tr, va) in enumerate(splits):
        bundle = fit_plsda(x[tr], y[tr], params['n_components'])
        oof[va] = predict_probability(bundle, x[va])
        score = metrics(y, oof)['balanced_accuracy']
        trial.report(float(score), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
    return float(metrics(y, oof)['balanced_accuracy'])


def train_torch_full(name, x, y, device, seed, epochs, params):
    """Fit one neural model on all available (possibly augmented) train rows."""
    seed_everything(seed)
    model = make_torch_model(name, dropout=params.get('dropout')).to(device)
    counts = np.bincount(y, minlength=2).astype(np.float32)
    pos_weight = torch.tensor([max(1.0, counts[0]) / max(1.0, counts[1])],
                              dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(params.get('lr', 1e-3)),
        weight_decay=float(params.get('weight_decay', 1e-3)))
    tx = tensor_for_model(name, x)
    ty = torch.tensor(y, dtype=torch.float32).view(-1, 1)
    batch = min(int(params.get('batch_size', 32)), max(1, len(ty)))
    loader = DataLoader(TensorDataset(tx, ty), batch_size=batch, shuffle=True,
                        drop_last=False)
    model.train()
    for _ in range(max(1, int(epochs))):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            if name == 'ResNetAE':
                logits, recon = model(bx)
                loss = criterion(logits, by) + float(params.get('recon_weight', 0.05)) * F.mse_loss(
                    recon, bx.flatten(1))
            else:
                loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    model.eval()
    return model


def train_pointnet_full(x, y, device, seed, epochs, params):
    seed_everything(seed)
    from leakage_free_pointnet_training import PointNetDual
    model = PointNetDual(num_classes=2, dropout=params.get('dropout')).to(device)
    counts = np.bincount(y, minlength=2).astype(np.float32)
    weights = torch.tensor(len(y) / (2.0 * np.maximum(counts, 1)),
                           dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(params.get('lr', 1e-3)),
        weight_decay=float(params.get('weight_decay', 1e-3)))
    tx = torch.tensor(x, dtype=torch.float32)
    ty = torch.tensor(y, dtype=torch.long)
    batch = min(int(params.get('batch_size', 32)), max(1, len(ty)))
    loader = DataLoader(TensorDataset(tx, ty), batch_size=batch, shuffle=True,
                        drop_last=False)
    model.train()
    for _ in range(max(1, int(epochs))):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
    model.eval()
    return model


def load_data(protocol, cohort, side, data_root=None):
    data_root = Path(data_root).resolve() if data_root else Path(__file__).resolve().parent
    if protocol in ('coef_raw', 'coef_plsda', 'plsda_direct'):
        train_df, test_df = load_raw_cohort(cohort, side, kind='coef', root=data_root)
        x, y, groups, columns = matrix(train_df)
        test_x, test_y, test_groups, _ = matrix(test_df)
    else:
        train_df, test_df = load_raw_xyz_cohort(cohort, side, root=data_root)
        x, y, groups = xyz_matrix(train_df)
        test_x, test_y, test_groups = xyz_matrix(test_df)
        columns = ['normalized_xyz_3x1002']
    if set(groups) & set(test_groups):
        raise ValueError(f'Patient overlap: {cohort}/{side}')
    return train_df, test_df, x, y, groups, columns, test_x, test_y, test_groups


def fit_final_and_test(protocol, model_name, params, train_df, test_df, x, y, groups,
                       columns, test_x, test_y, test_groups, splits, args, seed, run_dir):
    started = time.time()
    if protocol == 'plsda_direct':
        bundle = fit_plsda(x, y, params['n_components'])
        test_probability = predict_probability(bundle, test_x)
        with (run_dir / 'model.pkl').open('wb') as stream:
            pickle.dump(bundle, stream, protocol=pickle.HIGHEST_PROTOCOL)
    elif protocol in ('coef_raw', 'coef_plsda'):
        fit_x, fit_y, aug_info = coef_training_fold(
            x, y, seed, protocol, args.children_per_pair, args.noise_scale,
            args.pls_components)
        if model_name == 'SVM':
            model = svm_model(params, seed).fit(fit_x, fit_y)
            test_probability = model.predict_proba(test_x)[:, 1]
            with (run_dir / 'model.pkl').open('wb') as stream:
                pickle.dump(model, stream, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            scaler = StandardScaler().fit(fit_x)
            scaled_x = scaler.transform(fit_x).astype(np.float32)
            scaled_test = scaler.transform(test_x).astype(np.float32)
            model = train_torch_full(model_name, scaled_x, fit_y, args.device_obj,
                                     seed, args.final_epochs, params)
            with torch.no_grad():
                logits = model(tensor_for_model(model_name, scaled_test).to(args.device_obj))
                if model_name == 'ResNetAE':
                    logits = logits[0]
                test_probability = torch.sigmoid(logits).flatten().cpu().numpy()
            torch.save(model.state_dict(), run_dir / 'model_state.pt')
            with (run_dir / 'scaler.pkl').open('wb') as stream:
                pickle.dump(scaler, stream, protocol=pickle.HIGHEST_PROTOCOL)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    elif protocol in ('pointnet_raw', 'pointnet_plsda'):
        if protocol == 'pointnet_raw':
            fit_x, fit_y, aug_info = augment_xyz_fold(
                x, y, seed, 'balanced_jitter', args.noise_scale, args.children_per_pair)
        else:
            fit_x, fit_y, aug_info = plsda_augment_xyz_fold(
                x, y, seed, args.children_per_pair, args.pls_components)
        model = train_pointnet_full(fit_x, fit_y, args.device_obj, seed,
                                    args.final_epochs, params)
        with torch.no_grad():
            test_probability = torch.softmax(
                model(torch.tensor(test_x, dtype=torch.float32, device=args.device_obj)),
                dim=1)[:, 1].cpu().numpy()
        torch.save(model.state_dict(), run_dir / 'model_state.pt')
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        raise ValueError(protocol)

    test_metrics = metrics(test_y, test_probability)
    pd.DataFrame({
        'Subject': test_df['Subject'].astype(str), 'PatientID': test_groups,
        'BinaryClass': test_y, 'Probability': test_probability,
        'Prediction': (test_probability >= 0.5).astype(int),
    }).to_csv(run_dir / 'test_predictions.csv', index=False)
    manifest = {
        'protocol': f'optuna-{protocol}-v1',
        'source_protocol': RAW_COEF_PROTOCOL,
        'cohort': train_df['Subject'].astype(str).str.split('/').str[0].iloc[0]
                  if '/' in str(train_df['Subject'].iloc[0]) else '',
        'side': args.current_side, 'model': model_name,
        'feature_kind': 'coef' if protocol in ('coef_raw', 'coef_plsda', 'plsda_direct') else 'xyz',
        'optuna_best_params': jsonable(params), 'optuna_best_oof_balanced_accuracy': args.best_oof,
        'optuna_trials': args.n_trials, 'folds': len(splits), 'seed': int(seed),
        'test_evaluation': 'single_final_model_no_fold_ensemble',
        'train_rows': len(train_df), 'test_rows': len(test_df), 'features': columns,
        'data_root': str(args.data_root),
        'train_feature_sha256': digest_frame(train_df), 'test_feature_sha256': digest_frame(test_df),
        'patient_overlap': False, 'device': str(args.device_obj),
        'elapsed_seconds': round(time.time() - started, 2),
    }
    if protocol != 'plsda_direct':
        manifest['augmentation_applied_to_test'] = False
        manifest['augmentation_applied_to_validation'] = False
        manifest['children_per_pair'] = int(args.children_per_pair)
        if protocol == 'coef_plsda' or protocol == 'pointnet_plsda':
            manifest['pls_da_fit_scope'] = 'training_data_only'
            manifest['pls_components_requested'] = int(args.pls_components)
    (run_dir / 'run_manifest.json').write_text(json.dumps(manifest, indent=2),
                                                encoding='utf-8')
    (run_dir / 'metrics.json').write_text(json.dumps({'test': test_metrics}, indent=2),
                                           encoding='utf-8')
    return test_metrics


def study_one(protocol, model_name, cohort, side, args, output_root):
    train_df, test_df, x, y, groups, columns, test_x, test_y, test_groups = load_data(
        protocol, cohort, side, data_root=args.data_root)
    splits = grouped_splits(y, groups, args.folds, args.seed)
    study_dir = output_root / 'studies' / protocol / cohort / side / model_name
    study_dir.mkdir(parents=True, exist_ok=True)
    study_name = f'{protocol}_{cohort}_{side}_{model_name}'
    sampler = optuna.samplers.TPESampler(seed=args.seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=3)
    study = optuna.create_study(direction='maximize', sampler=sampler, pruner=pruner,
                                study_name=study_name)
    if protocol in ('coef_raw', 'coef_plsda'):
        objective = lambda trial: objective_coef(trial, model_name, protocol, x, y, splits, args)
    elif protocol in ('pointnet_raw', 'pointnet_plsda'):
        objective = lambda trial: objective_pointnet(trial, protocol, x, y, splits, args)
    elif protocol == 'plsda_direct':
        objective = lambda trial: objective_direct(trial, x, y, splits, args)
    else:
        raise ValueError(protocol)
    study.optimize(objective, n_trials=args.n_trials, n_jobs=1,
                   catch=(ValueError, RuntimeError, FloatingPointError))
    if study.best_trial is None:
        raise RuntimeError(f'No completed Optuna trial: {study_name}')
    best_params = jsonable(study.best_trial.params)
    study.trials_dataframe().to_csv(study_dir / 'trials.csv', index=False)
    (study_dir / 'best_params.json').write_text(json.dumps({
        'protocol': protocol, 'cohort': cohort, 'side': side, 'model': model_name,
        'folds': len(splits), 'seed': args.seed, 'n_trials_requested': args.n_trials,
        'n_trials_completed': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'best_trial': int(study.best_trial.number),
        'best_oof_balanced_accuracy': float(study.best_value),
        'best_params': best_params,
    }, indent=2), encoding='utf-8')

    final_rows = []
    args.best_oof = float(study.best_value)
    args.current_side = side
    for seed in args.eval_seeds:
        run_dir = output_root / 'final' / protocol / cohort / side / model_name / f'seed_{seed}'
        run_dir.mkdir(parents=True, exist_ok=True)
        test_metrics = fit_final_and_test(
            protocol, model_name, best_params, train_df, test_df, x, y, groups,
            columns, test_x, test_y, test_groups, splits, args, seed, run_dir)
        final_rows.append({
            'Protocol': PROTOCOL_LABELS[protocol], 'ProtocolKey': protocol,
            'Cohort': cohort, 'Side': side, 'Model': model_name,
            'EvalSeed': int(seed), 'OptunaTrials': int(args.n_trials),
            'BestOOF_BalancedAccuracy': float(study.best_value),
            'Test_Accuracy': test_metrics['accuracy'],
            'Test_BalancedAccuracy': test_metrics['balanced_accuracy'],
            'Test_Sensitivity': test_metrics['sensitivity'],
            'Test_Specificity': test_metrics['specificity'],
            'Test_F1_Macro': test_metrics['f1_macro'],
            'Test_ROC_AUC': test_metrics['roc_auc'],
            'BestParams': json.dumps(best_params, sort_keys=True),
            'RunDir': str(run_dir),
        })
    return final_rows, {
        'Protocol': PROTOCOL_LABELS[protocol], 'ProtocolKey': protocol,
        'Cohort': cohort, 'Side': side, 'Model': model_name,
        'OptunaTrials': int(args.n_trials), 'BestTrial': int(study.best_trial.number),
        'BestOOF_BalancedAccuracy': float(study.best_value),
        'BestParams': json.dumps(best_params, sort_keys=True),
        'StudyDir': str(study_dir),
    }


def allowed_models(protocol, requested):
    if requested != 'all':
        return (requested,)
    if protocol in ('coef_raw', 'coef_plsda'):
        return COEF_MODELS
    if protocol in ('pointnet_raw', 'pointnet_plsda'):
        return ('PointNet',)
    return ('PLSDA_classifier',)


def main():
    parser = argparse.ArgumentParser(description='Optuna grouped-OOF tuning for all model families')
    parser.add_argument('--protocol', choices=PROTOCOLS + ('all',), default='all')
    parser.add_argument('--cohort', choices=COHORTS + ('all',), default='all')
    parser.add_argument('--side', choices=SIDES + ('all',), default='all')
    parser.add_argument('--model', choices=COEF_MODELS + ('PointNet', 'PLSDA_classifier', 'all'), default='all')
    parser.add_argument('--folds', type=int, default=10)
    parser.add_argument('--n_trials', type=int, default=10)
    parser.add_argument('--tune_epochs', type=int, default=20)
    parser.add_argument('--tune_patience', type=int, default=5)
    parser.add_argument('--final_epochs', type=int, default=40)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval_seeds', default='42,123,2026')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--children_per_pair', type=int, default=8)
    parser.add_argument('--noise_scale', type=float, default=0.02)
    parser.add_argument('--pls_components', type=int, default=8)
    parser.add_argument('--data_root', default=str(Path(__file__).resolve().parent),
                        help='Root containing cohort-specific coefficient files and Output_Dataset XYZ files')
    parser.add_argument('--output_root', default=str(
        Path(__file__).resolve().parent / 'optuna_runs_10fold'))
    args = parser.parse_args()
    args.data_root = str(Path(args.data_root).resolve())
    args.eval_seeds = tuple(int(item.strip()) for item in str(args.eval_seeds).split(',') if item.strip())
    args.device_obj = torch.device(args.device if args.device != 'auto' else
                                   ('cuda' if torch.cuda.is_available() else 'cpu'))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    protocols = PROTOCOLS if args.protocol == 'all' else (args.protocol,)
    cohorts = COHORTS if args.cohort == 'all' else (args.cohort,)
    sides = SIDES if args.side == 'all' else (args.side,)
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    all_final, all_tuning = [], []
    failures = []
    for protocol in protocols:
        for cohort in cohorts:
            for side in sides:
                for model_name in allowed_models(protocol, args.model):
                    # ``--model`` is allowed to be all for one-family protocols,
                    # but silently skip incompatible names.
                    expected = allowed_models(protocol, 'all')
                    if model_name not in expected:
                        continue
                    try:
                        print(f'[{protocol}/{cohort}/{side}/{model_name}] '
                              f'10-fold Optuna trials={args.n_trials}', flush=True)
                        final_rows, tuning_row = study_one(
                            protocol, model_name, cohort, side, args, output_root)
                        all_final.extend(final_rows)
                        all_tuning.append(tuning_row)
                        print(f'  best OOF BA={tuning_row["BestOOF_BalancedAccuracy"]:.4f} '
                              f'params={tuning_row["BestParams"]}', flush=True)
                    except Exception as exc:
                        print(f'[FAIL] {protocol}/{cohort}/{side}/{model_name}: {exc}', flush=True)
                        failures.append({
                            'Protocol': PROTOCOL_LABELS.get(protocol, protocol),
                            'ProtocolKey': protocol, 'Cohort': cohort, 'Side': side,
                            'Model': model_name, 'Status': 'FAIL', 'Error': repr(exc),
                        })
    tuning_path = output_root / 'OPTUNA_TUNING_SUMMARY.csv'
    final_path = output_root / 'OPTUNA_FINAL_SUMMARY.csv'
    pd.DataFrame(all_tuning + failures).to_csv(tuning_path, index=False)
    pd.DataFrame(all_final + failures).to_csv(final_path, index=False)

    # Winners are selected by OOF score only.  Test values are attached for
    # reporting and are never used in this selection.
    tuning_df = pd.DataFrame(all_tuning)
    winner_rows = []
    if not tuning_df.empty:
        for _, group in tuning_df.groupby(['ProtocolKey', 'Cohort', 'Side'], sort=True):
            winner_rows.append(group.sort_values(
                ['BestOOF_BalancedAccuracy', 'Model'], ascending=[False, True]).iloc[0].to_dict())
    pd.DataFrame(winner_rows).to_csv(output_root / 'OPTUNA_WINNERS_BY_COHORT_SIDE.csv', index=False)
    (output_root / 'RUN_MANIFEST.json').write_text(json.dumps({
        'protocols': protocols, 'cohorts': cohorts, 'sides': sides,
        'data_root': args.data_root,
        'folds_requested': args.folds, 'trials_per_study': args.n_trials,
        'tuning_objective': 'grouped out-of-fold balanced accuracy',
        'test_used_during_tuning': False, 'test_evaluation': 'single model per seed',
        'eval_seeds': args.eval_seeds, 'device': str(args.device_obj),
        'failures': failures,
    }, indent=2), encoding='utf-8')
    print(f'Wrote {tuning_path}')
    print(f'Wrote {final_path}')
    print(f'Wrote {output_root / "OPTUNA_WINNERS_BY_COHORT_SIDE.csv"}')
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()

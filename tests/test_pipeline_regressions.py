"""Regression tests with real tabular data and isolated external-app boundaries.

Slicer/Qt/torch MRI execution is intentionally not simulated as an end-to-end test.
Run: PythonSlicer.exe -m unittest discover -s tests -v
"""
import argparse
import ast
import contextlib
import glob
import io
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'Model'))
from Model.data_contract import load_cohort, canonical_frame, cohort_identity, PROTOCOL
from Model.evaluation import compute_bootstrap_stats
from Model.validated_model import FoldAugmentedSVC
from Model.compare_runs import compare
from Data_Processing.feature_metadata import load_labels, write_feature_manifest
from ICP.reference_contract import write_reference_contract, load_reference_contract


def extract(path, names, ns):
    tree = ast.parse((ROOT / path).read_text(encoding='utf-8-sig'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(nodes) != len(names):
        raise AssertionError((path, names))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), path, 'exec'), ns)
    return ns


class PredictorSchemaTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse((ROOT / 'DesktopApp/models/predictor.py').read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'HippocampalPredictor')
        node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_prepare_input')
        ns = {'np': np, 'pd': pd}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'predictor.py', 'exec'), ns)
        self.prepare = ns['_prepare_input']
        self.subject = SimpleNamespace(scalers={'left': SimpleNamespace(feature_names_in_=np.array(['Coef_1', 'Coef_2']))}, pipelines={})

    def test_reordered_named_columns_and_series_preserve_values(self):
        frame = pd.DataFrame({'Coef_2': [20.0], 'Coef_1': [10.0]})
        for item in (frame, frame.iloc[0]):
            actual = self.prepare(self.subject, item, 'left')
            self.assertEqual(actual.iloc[0].to_dict(), {'Coef_1': 10.0, 'Coef_2': 20.0})

    def test_missing_extra_and_duplicate_columns_rejected(self):
        for frame in (pd.DataFrame({'Coef_1': [1]}), pd.DataFrame({'Coef_1': [1], 'wrong': [2]}), pd.DataFrame([[1, 2]], columns=['Coef_1', 'Coef_1'])):
            with self.assertRaises(ValueError):
                self.prepare(self.subject, frame, 'left')

    def test_nonfinite_and_wrong_width_rejected(self):
        for arr in (np.array([[np.nan, 1]]), np.array([[1, np.inf]]), np.array([1, 2, 3]), np.zeros((0, 2))):
            with self.assertRaises(ValueError):
                self.prepare(self.subject, arr, 'left')

    def test_new_features_require_compatible_model_before_loading_weights(self):
        tree = ast.parse((ROOT / 'DesktopApp/models/predictor.py').read_text(encoding='utf-8'))
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'HippocampalPredictor')
        node = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == 'predict')
        ns = {'os': os, 're': re, 'json': json}
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'predictor.py', 'exec'), ns)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td, 'case_SPHARM.coef')
            Path(td, 'case_processing.json').write_text(json.dumps({'feature_contract': {'icp_mode': 'fixed_reference'}}))
            subject = SimpleNamespace(models_root=td)
            with self.assertRaisesRegex(ValueError, 'manifest'):
                ns['predict'](subject, str(path), 'left')


class PipelineStatusTests(unittest.TestCase):
    def test_paired_mri_saved_and_copy_failure_not_success(self):
        ns = extract('run_pipeline.py', ['organize_output'], {'os': os, 'np': np, 'shutil': shutil})
        with tempfile.TemporaryDirectory() as td:
            ns['OUTPUT_DIR'] = td
            for name in ('lh_demo_hippocampus.nii.gz', 'rh_demo_hippocampus.nii.gz', 'orig.mgz'):
                Path(td, name).write_bytes(b'fixture')
            nib = SimpleNamespace(load=lambda _: SimpleNamespace(dataobj=[1], affine=np.eye(4)),
                                  Nifti1Image=lambda array, affine: (array, affine),
                                  save=lambda image, path: Path(path).write_bytes(b'saved'))
            with patch.dict(sys.modules, {'nibabel': nib}), contextlib.redirect_stdout(io.StringIO()):
                self.assertTrue(ns['organize_output'](td, 'demo', str(Path(td, 'orig.mgz'))))
                self.assertTrue(Path(td, 'mri/demo_t1.nii.gz').is_file())
                nib.save = lambda *a: (_ for _ in ()).throw(OSError('disk failure'))
                self.assertFalse(ns['organize_output'](td, 'demo', str(Path(td, 'orig.mgz'))))

    def test_all_segmentation_failures_return_nonzero(self):
        with tempfile.TemporaryDirectory() as td:
            ns = extract('run_pipeline.py', ['main', 'get_subject_id'], {'os': os, 'sys': sys, 'glob': glob, 'argparse': argparse,
                'time': __import__('time'), 'shutil': shutil, 'datetime': __import__('datetime').datetime,
                'OUTPUT_DIR': str(Path(td, 'out')), 'FASTSURFER_OUTPUT_DIR': str(Path(td, 'fastsurfer')),
                'PIPELINE_DIR': td, 'EXTRACTION_MODE': 'moderate', 'print_banner': lambda: None,
                'validate_setup': lambda **k: None, 'run_fastsurfer': lambda *a: None})
            with patch.object(sys, 'argv', ['run_pipeline.py', '--input', 'demo.nii.gz']), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(ns['main'](), 1)

    def test_duplicate_subject_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            ns = extract('run_pipeline.py', ['main', 'get_subject_id'], {'os': os, 'sys': sys, 'argparse': argparse,
                'OUTPUT_DIR': td, 'FASTSURFER_OUTPUT_DIR': td,
                'print_banner': lambda: None, 'validate_setup': lambda **k: None})
            with patch.object(sys, 'argv', ['run_pipeline.py', '--input', 'datasetA/sub-01.nii.gz', 'datasetB/sub-01.nii.gz']):
                with self.assertRaisesRegex(ValueError, 'collide'):
                    ns['main']()

    def test_spharm_subject_exception_returns_false(self):
        ns = extract('SPHARM/run_spharm_batch.py', ['process_single_subject'], {'os': os, 'glob': glob, 'sprint': lambda *a: None,
            'cleanup_subject_files': lambda *a: 0,
            'slicer': SimpleNamespace(util=SimpleNamespace(loadLabelVolume=lambda _: (_ for _ in ()).throw(RuntimeError('load failure'))))})
        with tempfile.TemporaryDirectory() as td:
            source = Path(td, 'demo.nii.gz')
            source.write_bytes(b'fixture')
            args = SimpleNamespace(regenerate_spharm_only=False, input_dir=td)
            self.assertFalse(ns['process_single_subject'](str(source), 0, 1, td, args, {'degree': 12, 'subdiv': 10}, None, None))
            self.assertFalse(Path(td, 'demo_processing.json').exists())

    def test_spharm_all_failed_batch_raises_and_records_count(self):
        with tempfile.TemporaryDirectory() as td:
            args = SimpleNamespace(num_iterations=None, subdiv_level=None, spharm_degree=None, fast=False, output_dir=td, input_dir=td, num_shards=1, shard_index=0)
            ns = extract('SPHARM/run_spharm_batch.py', ['run_batch_spharm'], {'os': os, 'SCRIPT_DIR': td, 'parse_args': lambda: args,
                'init_logging': lambda *a, **k: None, 'find_label_files': lambda *a: ['a.nii.gz', 'b.nii.gz'],
                'resolve_reference_template': lambda *a: None, 'process_single_subject': lambda *a: False, 'sprint': lambda *a: None})
            with self.assertRaisesRegex(RuntimeError, '2/2'):
                ns['run_batch_spharm']()
            self.assertEqual(json.loads(Path(td, 'spharm_status_shard0.json').read_text())['failed'], 2)

    def test_cli_completed_with_errors_is_failure(self):
        node = SimpleNamespace(GetStatusString=lambda: 'Completed with errors', GetErrorText=lambda: 'failed', GetOutputText=lambda: '')
        ns = extract('SPHARM/run_spharm_batch.py', ['run_cli_checked'], {'slicer': SimpleNamespace(cli=SimpleNamespace(run=lambda *a, **k: node)), 'sprint': lambda *a: None})
        self.assertFalse(ns['run_cli_checked'](None, {}, 'test', None))


class ReferenceTests(unittest.TestCase):
    def test_reference_metadata_preserves_scale_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as td:
            template, source = Path(td, 'mean.ply'), Path(td, 'train.nii.gz')
            template.write_bytes(b'template')
            source.write_bytes(b'mask')
            write_reference_contract(template, 0.04, [source], 0.02, 128)
            self.assertEqual(load_reference_contract(template)['physical_to_normalized_scale'], 0.04)
            template.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                load_reference_contract(template)

    def test_legacy_template_without_physical_scale_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td, 'mean.vtk')
            path.write_bytes(b'template')
            with self.assertRaisesRegex(ValueError, 'metadata'):
                load_reference_contract(path)

    def test_reference_argument_parsed(self):
        ns = extract('ICP/ICP.py', ['parse_args'], dict(argparse=argparse, OUTPUT_VOXELS=128, MAX_GW_ITERATIONS=20, GW_TOLERANCE=.001,
             PAIRWISE_ITERATIONS=100, PAIRWISE_TOLERANCE=.001, PAIRWISE_LANDMARKS=200, INTERPOLATION_MODE='nn', os=os))
        with patch.object(sys, 'argv', ['ICP.py', '--reference_template', 'ref.ply']):
            self.assertEqual(ns['parse_args']().reference_template, 'ref.ply')

    def test_exploratory_groupwise_requires_explicit_opt_in(self):
        ns = extract('ICP/ICP.py', ['parse_args'], dict(argparse=argparse, OUTPUT_VOXELS=128, MAX_GW_ITERATIONS=20, GW_TOLERANCE=.001,
             PAIRWISE_ITERATIONS=100, PAIRWISE_TOLERANCE=.001, PAIRWISE_LANDMARKS=200, INTERPOLATION_MODE='nn', os=os))
        with patch.object(sys, 'argv', ['ICP.py']):
            with self.assertRaises(SystemExit):
                ns['parse_args']()
        with patch.object(sys, 'argv', ['ICP.py', '--exploratory_groupwise']):
            self.assertTrue(ns['parse_args']().exploratory_groupwise)

    def test_single_file_input_requires_existing_volume_and_output(self):
        ns = extract('ICP/ICP.py', ['parse_args'], dict(argparse=argparse, OUTPUT_VOXELS=128, MAX_GW_ITERATIONS=20, GW_TOLERANCE=.001,
             PAIRWISE_ITERATIONS=100, PAIRWISE_TOLERANCE=.001, PAIRWISE_LANDMARKS=200, INTERPOLATION_MODE='nn', os=os))
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir, 'one_left_mask.nii.gz')
            path.write_bytes(b'test')
            args = ['ICP.py', '--reference_template', 'ref.ply', '--input_file', str(path),
                    '--output_dir', str(Path(temp_dir, 'output'))]
            with patch.object(sys, 'argv', args):
                parsed = ns['parse_args']()
            self.assertEqual(parsed.input_file, str(path))

    def test_fixed_alignment_not_affected_by_other_subjects(self):
        class Mesh:
            def __init__(self, pts): self.pts = np.asarray(pts, dtype=float)
            def GetNumberOfPoints(self): return len(self.pts)
            def GetBounds(self): return tuple(np.column_stack((self.pts.min(0), self.pts.max(0))).ravel())
        reference = Mesh(np.zeros((12, 3)))
        reader = SimpleNamespace(SetFileName=lambda _: None, Update=lambda: None, GetOutput=lambda: reference)
        contract = {'physical_to_normalized_scale': .02, 'output_spacing': .02, 'output_voxels': 128}
        ns = extract('ICP/ICP.py', ['align_to_fixed_reference'], {'np': np, 'os': os,
            'load_reference_contract': lambda _: contract, 'vtk': SimpleNamespace(vtkPLYReader=lambda: reader),
            'get_points_numpy': lambda mesh: mesh.pts,
            'apply_poly_transform': lambda mesh, m: Mesh(mesh.pts @ m[:3, :3].T + m[:3, 3]),
            'principal_axis_align': lambda mesh: (mesh, np.eye(4)),
            'pole_flip_correction': lambda mesh, **k: (mesh, np.eye(4), False),
            'run_vtk_icp': lambda *a, **k: np.eye(4)})
        args = SimpleNamespace(reference_template='ref.ply', pairwise_iterations=100, pairwise_tolerance=.001, pairwise_landmarks=100)
        mesh = Mesh([[x,y,z] for x in (-2,2) for y in (-3,3) for z in (-4,4)])
        single, matrices1, _ = ns['align_to_fixed_reference']([mesh], ['a'], args)
        batch, matrices2, _ = ns['align_to_fixed_reference']([mesh, Mesh(mesh.pts * 2)], ['a','b'], args)
        np.testing.assert_allclose(single[0].pts, batch[0].pts)
        np.testing.assert_allclose(matrices1[0], matrices2[0])
        self.assertAlmostEqual(np.linalg.det(matrices1[0][:3,:3]), .02**3)


class EvaluationTests(unittest.TestCase):
    def test_benchmark_rejects_different_subjects_and_accepts_reordering(self):
        with tempfile.TemporaryDirectory() as td:
            paths = [Path(td, f'{i}.npz') for i in range(3)]
            for path, ids, truth in zip(paths, (['a','b'], ['b','a'], ['a','c']), ([0,1], [1,0], [0,1])):
                np.savez(path, subject_ids=ids, y_test=truth, y_pred=truth, y_prob=truth, protocol=PROTOCOL)
            self.assertEqual(len(compare(paths[:2])), 2)
            with self.assertRaisesRegex(ValueError, 'different test'):
                compare([paths[0], paths[2]])

    def test_tied_scores_have_auc_half_in_all_valid_resamples(self):
        out = compute_bootstrap_stats([0,0,1,1], [.5]*4, n_boot=200)
        self.assertEqual(out['Mean_AUC'], '0.5000')
        self.assertEqual(out['AUC_95CI'], '[0.5000, 0.5000]')
        self.assertGreater(out['Bootstrap_Skipped'], 0)

    def test_saved_classifier_decisions_used_instead_of_probability_threshold(self):
        out = compute_bootstrap_stats([0,0,1,1], [.9,.9,.1,.1], n_boot=50, y_pred=[0,0,1,1])
        self.assertEqual(out['Mean_Acc'], '100.00%')
        self.assertEqual(out['Decision_Source'], 'saved_y_pred')

    def test_invalid_labels_probabilities_and_single_class_rejected(self):
        for truth, prob in (([0,-1],[.1,.5]), ([0,1],[np.nan,.5]), ([0,0],[.1,.2])):
            with self.assertRaises(ValueError):
                compute_bootstrap_stats(truth, prob)

    def test_seed_is_reproducible_without_mutating_global_rng(self):
        np.random.seed(7)
        expected = np.random.random()
        np.random.seed(7)
        a = compute_bootstrap_stats([0,0,1,1], [.2,.4,.6,.8], n_boot=50)
        self.assertEqual(expected, np.random.random())
        self.assertEqual(a, compute_bootstrap_stats([0,0,1,1], [.2,.4,.6,.8], n_boot=50))


class DataContractTests(unittest.TestCase):
    def test_archived_reports_stop_before_overwriting_historical_evidence(self):
        folder = ROOT / 'Model_Results_Excel' / '04_Scripts'
        for name in ('generate_model_excel.py', 'generate_newest_bootstrap_results.py'):
            tree = ast.parse((folder / name).read_text(encoding='utf-8-sig'))
            main = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == 'main')
            self.assertIsInstance(main.body[0], ast.Raise)
            namespace = {}
            exec(compile(ast.Module(body=[main], type_ignores=[]), name, 'exec'), namespace)
            with self.assertRaisesRegex(RuntimeError, 'compare_runs.py'):
                namespace['main']()

    def test_explicit_label_metadata_overrides_filename_heuristics(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td, 'labels.csv')
            path.write_text('Subject,BinaryClass,Class,Group\nsub-001,1,1,TLE\n', encoding='utf-8')
            self.assertEqual(load_labels(path)['sub-001'], ('TLE', 1, 1))
            path.write_text('Subject,BinaryClass,Class\nsub-001,0,1\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                load_labels(path)

    def test_mixed_feature_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            files = [Path(td, f'{i}_SPHARM.coef') for i in range(2)]
            csv_path = Path(td, 'features.csv')
            csv_path.write_text('fixture')
            for i in range(2):
                Path(td, f'{i}_processing.json').write_text(json.dumps({'feature_contract': {'template': i}}))
            with self.assertRaisesRegex(ValueError, 'Mixed preprocessing'):
                write_feature_manifest(csv_path, files)

    def test_real_cohorts_match_xyz_coefficients_and_have_disjoint_subjects(self):
        for cohort in ('Ds004469','Ds005602','All_Augment_tain','combind_methode','Ds004469Train_Ds005602test','Ds005602Train_Ds004469test'):
            for side in ('left','right'):
                with self.subTest(cohort=cohort, side=side):
                    coef, xyz = load_cohort(cohort, side, 'coef'), load_cohort(cohort, side, 'xyz')
                    for a, b in zip(coef, xyz):
                        self.assertEqual(cohort_identity(a), cohort_identity(b))
                    self.assertFalse(set(coef[0].Subject) & set(coef[1].Subject))

    def test_unknown_and_preaugmented_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td, 'data.csv')
            for name, label in [('sub1', -1), ('TLE_aug_001', 1)]:
                pd.DataFrame({'Subject':[name], 'Class':[label], 'BinaryClass':[int(label == 1)], 'Coef_1':[1.0]}).to_csv(path, index=False)
                with self.assertRaises(ValueError):
                    canonical_frame(path, 'fixture')

    def test_augmentation_is_local_to_fit_and_same_class(self):
        rng = np.random.RandomState(5)
        X = rng.normal(size=(24, 8))
        y = np.array([0]*12 + [1]*12)
        model = FoldAugmentedSVC(n_components=2, children_per_subject=2).fit(X[:20], y[:20])
        self.assertEqual(model.n_original_fit_, 20)
        self.assertEqual(model.n_synthetic_fit_, 40)
        np.testing.assert_allclose(model.scaler_.mean_, X[:20].mean(0))
        old_mean = model.scaler_.mean_.copy()
        model.predict(X[20:] + 1000)
        np.testing.assert_array_equal(model.scaler_.mean_, old_mean)

    def test_legacy_training_entry_points_are_removed_and_current_runners_parse(self):
        legacy = [
            path for path in (ROOT / 'Model').rglob('train_*.py')
            if 'Backup' not in path.parts and path.parent != ROOT / 'Model'
        ]
        self.assertEqual(legacy, [])
        current = [
            ROOT / 'Model' / 'leakage_free_coef_training.py',
            ROOT / 'Model' / 'leakage_free_plsda_training.py',
            ROOT / 'Model' / 'leakage_free_pointnet_training.py',
            ROOT / 'Model' / 'leakage_free_pointnet_plsda_training.py',
            ROOT / 'Model' / 'optuna_leakage_free_all.py',
        ]
        for path in current:
            ast.parse(path.read_text(encoding='utf-8-sig'))


if __name__ == '__main__':
    unittest.main()

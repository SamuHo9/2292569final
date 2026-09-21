import csv
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from ICP.prepare_mask_split import prepare as prepare_mask_split
from ICP.reference_contract import LEGACY_GROUPWISE_VERSION, load_reference_contract
from Model.build_fixed_reference_inputs import build as build_model_inputs
from Model.data_contract import load_raw_cohort, load_raw_xyz_cohort


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PreIcpMaskSplitTests(unittest.TestCase):
    def make_fixture(self, root):
        manifest_path = root / 'current_split_manifest.json'
        left_status = root / 'left_status.csv'
        right_status = root / 'right_status.csv'
        left_source = root / 'left_source'
        right_source = root / 'right_source'
        left_source.mkdir()
        right_source.mkdir()
        cases = [
            ('sub-0001', 'train', 'Healthy', 'Ds004469'),
            ('sub-0002', 'train', 'TLE', 'Ds005602'),
            ('sub-0003', 'test', 'Healthy', 'Ds004469'),
            ('sub-0004', 'test', 'TLE', 'Ds005602'),
        ]
        subjects = []
        status_rows = {'left': [], 'right': []}
        for sid, split, label, dataset in cases:
            subjects.append({
                'subject_id': sid, 'split': split,
                'left_label': label, 'right_label': label,
            })
            for side, prefix, hemi in (('left', 'lh', 'lh'), ('right', 'rh', 'rh')):
                status_rows[side].append({
                    'FileName': f"{side}_{label}_{sid}_hippocampus_{hemi}_aligned_SPHARM.coef",
                    'Status': label, 'Dataset': dataset,
                })
                source = left_source if side == 'left' else right_source
                (source / f'{prefix}_{sid}_hippocampus.nii.gz').write_bytes(b'mask-' + sid.encode())
        # A raw mask with no audited left label must be reported and excluded.
        subjects.append({'subject_id': 'sub-0005', 'split': 'train',
                         'left_label': None, 'right_label': None})
        (left_source / 'lh_sub-0005_hippocampus.nii.gz').write_bytes(b'excluded-mask')
        manifest_path.write_text(json.dumps({
            'policy': 'patient_subject_level_stratified_split', 'subjects': subjects,
        }), encoding='utf-8')
        for path, side in ((left_status, 'left'), (right_status, 'right')):
            with path.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=['FileName', 'Status', 'Dataset'])
                writer.writeheader()
                writer.writerows(status_rows[side])
        return left_source, right_source, manifest_path, left_status, right_status

    def test_stages_copies_with_shared_subject_split_and_explicit_labels(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            left, right, manifest, left_status, right_status = self.make_fixture(root)
            output = root / 'fresh_run'
            result = prepare_mask_split(left, right, manifest, left_status, right_status, output)
            self.assertEqual(result['side_counts']['left']['train'], 2)
            self.assertEqual(result['side_counts']['left']['test'], 2)
            self.assertEqual(result['side_counts']['left']['excluded_unstatused'], 1)
            self.assertTrue((left / 'lh_sub-0001_hippocampus.nii.gz').is_file())
            self.assertTrue((output / 'masks' / 'left' / 'train' / 'lh_sub-0001_hippocampus.nii.gz').is_file())
            label_path = output / 'labels' / 'left' / 'train_labels.csv'
            with label_path.open(newline='', encoding='utf-8') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({row['PatientID'] for row in rows}, {'sub-0001', 'sub-0002'})
            tle = next(row for row in rows if row['PatientID'] == 'sub-0002')
            self.assertEqual(tle['Subject'], 'lh_sub-0002_hippocampus_aligned')
            self.assertEqual((tle['Class'], tle['BinaryClass'], tle['Dataset']), ('1', '1', 'Ds005602'))
            self.assertEqual(file_hash(left / 'lh_sub-0001_hippocampus.nii.gz'),
                             file_hash(output / 'masks' / 'left' / 'train' / 'lh_sub-0001_hippocampus.nii.gz'))

    def test_missing_required_mask_fails_before_staging(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            left, right, manifest, left_status, right_status = self.make_fixture(root)
            (right / 'rh_sub-0004_hippocampus.nii.gz').unlink()
            output = root / 'fresh_run'
            with self.assertRaisesRegex(FileNotFoundError, 'Missing right'):
                prepare_mask_split(left, right, manifest, left_status, right_status, output)
            self.assertFalse(output.exists())


class LegacyIcpReferenceBundleTests(unittest.TestCase):
    def test_recovered_references_are_hash_valid_and_marked_non_independent(self):
        project_root = Path(__file__).resolve().parents[1]
        bundle_root = project_root / 'ICP' / 'references' / 'legacy_groupwise_all_381_v1'
        manifest = json.loads((bundle_root / 'reference_manifest.json').read_text(encoding='utf-8'))
        self.assertFalse(manifest['independent_test_reference'])
        self.assertEqual(manifest['reference_version'], LEGACY_GROUPWISE_VERSION)
        self.assertEqual(manifest['sides']['left']['subject_count'], 381)
        self.assertEqual(manifest['sides']['right']['subject_count'], 381)
        self.assertEqual(manifest['current_dataset_overlap']['left']['current_subjects_in_legacy_reference'], 373)
        self.assertEqual(manifest['current_dataset_overlap']['right']['current_subjects_in_legacy_reference'], 377)
        self.assertTrue(manifest['current_dataset_overlap']['left']['all_current_subjects_are_in_legacy_reference'])
        self.assertTrue(manifest['current_dataset_overlap']['right']['all_current_subjects_are_in_legacy_reference'])
        for side in ('left', 'right'):
            template = bundle_root / side / 'mean_shape.ply'
            contract = load_reference_contract(template)
            self.assertEqual(contract['version'], LEGACY_GROUPWISE_VERSION)
            self.assertFalse(contract['independent_test_reference'])
            self.assertEqual(contract['output_voxels'], 128)
            self.assertAlmostEqual(contract['output_spacing'], 0.015625)
            self.assertGreater(contract['physical_to_normalized_scale'], 0)


class FixedReferenceModelInputTests(unittest.TestCase):
    def write_feature_bundle(self, path, kind, side, split):
        if kind == 'coef':
            feature_names = [f'Coef_{i}' for i in range(1, 508)]
            header = ['Subject', 'Group', 'Class', 'BinaryClass', 'Dataset'] + feature_names
        else:
            feature_names = [f'{axis}_{i}' for i in range(1002) for axis in 'xyz']
            header = ['Subject', 'Group_Name', 'Group_Label', 'BinaryClass', 'Dataset'] + feature_names
        rows = []
        id_base = 10000 if split == 'train' else 20000
        for dataset, dataset_offset in (('Ds004469', 0), ('Ds005602', 1000)):
            for binary in (0, 1):
                sid = f'sub-{id_base + dataset_offset + binary}'
                prefix = 'lh' if side == 'left' else 'rh'
                subject = f'{prefix}_{sid}_hippocampus_aligned'
                group = 'Non-affected/control' if binary == 0 else 'Ipsilateral TLE (Diseased)'
                sample_offset = id_base + dataset_offset + binary
                values = {
                    'Subject': subject, 'Dataset': dataset, 'BinaryClass': binary,
                    **({'Group': group, 'Class': binary} if kind == 'coef' else
                       {'Group_Name': group, 'Group_Label': binary}),
                }
                values.update({name: f'{binary + i / 10000 + sample_offset / 1000000:.8f}'
                               for i, name in enumerate(feature_names)})
                rows.append(values)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        processing = {
            'icp_mode': 'fixed_reference', 'icp_reference_sha256': f'icp-{side}-hash',
            'spharm_template_sha256': f'spharm-{side}-hash', 'spharm_degree': 12,
            'subdiv_level': 10, 'grid_theta': 4.5, 'grid_phi': 4.5,
        }
        contract = ({'coefficient_variant': 'ellalign', **processing} if kind == 'coef' else {
            'mesh_variant': 'ellalign', 'num_points': 1002, 'point_order': 'x,y,z',
            'processing_provenance': processing,
        })
        Path(str(path) + '.json').write_text(json.dumps({
            'csv_sha256': file_hash(path), 'feature_contract': contract,
        }), encoding='utf-8')

    def test_built_root_loads_in_coefficient_and_pointnet_loaders(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            run_root = root / 'rerun'
            for side in ('left', 'right'):
                for split in ('train', 'test'):
                    feature_root = run_root / 'icp' / side / f'{split}_fixed' / 'ml_features'
                    self.write_feature_bundle(feature_root / 'spharm_results_coef_features.csv', 'coef', side, split)
                    self.write_feature_bundle(feature_root / 'spharm_xyz_coords.csv', 'xyz', side, split)
            output = run_root / 'model_inputs'
            with contextlib.redirect_stdout(io.StringIO()):
                build_model_inputs(run_root, output)
            for cohort in ('All_coef', 'All_coef_Ds004469', 'All_coef_Ds005602'):
                for side in ('left', 'right'):
                    coef_train, coef_test = load_raw_cohort(cohort, side, root=output)
                    xyz_train, xyz_test = load_raw_xyz_cohort(cohort, side, root=output)
                    self.assertEqual(len(coef_train), 4 if cohort == 'All_coef' else 2)
                    self.assertEqual(len(coef_test), 4 if cohort == 'All_coef' else 2)
                    self.assertEqual(len(xyz_train), len(coef_train))
                    self.assertEqual(len(xyz_test), len(coef_test))
                    self.assertEqual(set(coef_train.BinaryClass), {0, 1})
                    self.assertEqual(set(coef_test.BinaryClass), {0, 1})


if __name__ == '__main__':
    unittest.main()

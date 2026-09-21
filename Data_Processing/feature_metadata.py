"""Record coefficient preprocessing provenance and explicit subject labels."""
import csv
import hashlib
import json
from pathlib import Path
import re


def load_labels(path):
    if not path:
        return None
    with open(path, newline='', encoding='utf-8-sig') as stream:
        rows = csv.DictReader(stream)
        if not {'Subject', 'BinaryClass'}.issubset(rows.fieldnames or []):
            raise ValueError('Label CSV requires Subject and BinaryClass columns')
        labels = {}
        for row in rows:
            subject = row['Subject'].strip()
            binary = int(row['BinaryClass'])
            cls = int(row.get('Class') or binary)
            if not subject or subject in labels or binary not in (0, 1) or cls not in (0, 1, 2) or int(cls == 1) != binary:
                raise ValueError('Invalid, duplicate, or inconsistent subject label')
            labels[subject] = (row.get('Group') or f'Class {binary}', cls, binary)
    return labels


def load_dataset_labels(path):
    """Load optional cohort membership from the explicit labels CSV."""
    if not path:
        return None
    with open(path, newline='', encoding='utf-8-sig') as stream:
        rows = csv.DictReader(stream)
        if 'Subject' not in (rows.fieldnames or []):
            raise ValueError('Label CSV requires a Subject column')
        if 'Dataset' not in (rows.fieldnames or []):
            return None
        result = {}
        for row in rows:
            subject = str(row.get('Subject') or '').strip()
            dataset = str(row.get('Dataset') or '').strip()
            if not subject or not dataset or subject in result:
                raise ValueError('Dataset labels must have unique, nonempty Subject and Dataset values')
            if dataset not in {'Ds004469', 'Ds005602'}:
                raise ValueError(f'Unexpected source Dataset for {subject}: {dataset!r}')
            result[subject] = dataset
    return result


def validate_feature_contracts(coef_files):
    contracts = []
    for path in coef_files:
        base = re.sub(r'_SPHARM(?:_ellalign|_procalign)?\.coef$', '', str(path))
        stamp = Path(base + '_processing.json')
        if not stamp.is_file():
            raise ValueError(f'Missing preprocessing provenance sidecar: {stamp}')
        payload = json.loads(stamp.read_text(encoding='utf-8'))
        contract = payload.get('feature_contract')
        if not isinstance(contract, dict):
            raise ValueError(f'Missing feature contract in: {stamp}')
        contracts.append(contract)
    unique = {json.dumps(c, sort_keys=True) for c in contracts}
    if len(unique) != 1:
        raise ValueError('Mixed preprocessing contracts; do not combine these coefficients')
    return contracts[0]


def write_feature_manifest(csv_path, coef_files, label_path=None, contract=None):
    """Write a feature manifest.

    By default the coefficient sidecars are validated strictly.  The optional
    explicit contract is reserved for the extractor's opt-in exploratory mode
    when historical or synthetic coefficients have no provenance sidecar.
    """
    if contract is None:
        contract = validate_feature_contracts(coef_files)
    payload = {'csv_sha256': hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
               'feature_contract': contract,
               'label_source': str(label_path) if label_path else 'filename_heuristic; unknown=-1',
               'label_sha256': hashlib.sha256(Path(label_path).read_bytes()).hexdigest() if label_path else None,
               'clinical_validation': False}
    Path(str(csv_path) + '.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')


def write_geometry_manifest(csv_path, mesh_files, mesh_variant, num_points,
                            point_order='x,y,z', label_path=None,
                            processing_contract=None):
    """Write the contract for an XYZ mesh feature CSV.

    Keeping this next to the coefficient manifest makes train/test/balanced
    geometry files auditable and lets the model loader reject mixed mesh
    variants before training.
    """
    payload = {
        'csv_sha256': hashlib.sha256(Path(csv_path).read_bytes()).hexdigest(),
        'feature_contract': {
            'version': 'spharm-geometry-v1',
            'mesh_variant': mesh_variant,
            'num_points': int(num_points),
            'point_order': point_order,
            'processing_provenance': processing_contract,
        },
        'source_count': len(mesh_files),
        'label_source': str(label_path) if label_path else 'filename_heuristic; unknown=-1',
        'label_sha256': hashlib.sha256(Path(label_path).read_bytes()).hexdigest() if label_path else None,
        'clinical_validation': False,
    }
    Path(str(csv_path) + '.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')


def validate_mesh_processing_contracts(mesh_files):
    """Require one identical per-hemisphere ICP/SPHARM contract for a mesh batch."""
    contracts = []
    suffixes = (
        '_SPHARM_ellalign.vtk', '_SPHARM_procalign.vtk',
        '_SPHARM_realigned.vtk', '_SPHARM.vtk',
    )
    for mesh in mesh_files:
        path = Path(mesh)
        stem = path.name
        for suffix in suffixes:
            if stem.endswith(suffix):
                stem = stem[:-len(suffix)]
                break
        sidecar = path.parent / f'{stem}_processing.json'
        if not sidecar.is_file():
            raise ValueError(f'Missing preprocessing provenance sidecar: {sidecar}')
        try:
            payload = json.loads(sidecar.read_text(encoding='utf-8'))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f'Invalid preprocessing provenance sidecar: {sidecar}') from exc
        contract = payload.get('feature_contract')
        if not isinstance(contract, dict):
            raise ValueError(f'Missing feature_contract in: {sidecar}')
        contracts.append(contract)
    unique = {json.dumps(contract, sort_keys=True) for contract in contracts}
    if len(unique) != 1:
        raise ValueError('Mixed ICP/SPHARM preprocessing contracts; do not combine these meshes')
    return contracts[0]

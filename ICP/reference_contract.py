"""Persist the physical scale with a training reference; never infer it from a new batch."""
import hashlib
import json
from pathlib import Path
import numpy as np

VERSION = 'fixed-training-reference-v1'
LEGACY_GROUPWISE_VERSION = 'fixed-legacy-groupwise-reference-v1'
SUPPORTED_VERSIONS = {VERSION, LEGACY_GROUPWISE_VERSION}


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_reference_contract(template, physical_scale, subject_paths, spacing, voxels):
    scale = float(physical_scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Reference scale must be positive and finite')
    payload = {'version': VERSION, 'template_sha256': file_sha256(template),
               'physical_to_normalized_scale': scale,
               'output_spacing': float(spacing), 'output_voxels': int(voxels),
               'training_inputs': [{'name': Path(p).name, 'sha256': file_sha256(p)} for p in subject_paths]}
    Path(str(template) + '.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    return payload


def load_reference_contract(template):
    path = Path(str(template) + '.json')
    if not path.is_file():
        raise ValueError(f'Missing training-reference metadata: {path}. Build with ICP.py --fit_reference using training masks only. Legacy normalized templates cannot supply the original physical scale.')
    payload = json.loads(path.read_text(encoding='utf-8'))
    if payload.get('version') not in SUPPORTED_VERSIONS or payload.get('template_sha256') != file_sha256(template):
        raise ValueError('Reference template/metadata mismatch')
    scale = payload.get('physical_to_normalized_scale', 0)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Invalid physical reference scale')
    if payload.get('output_spacing', 0) <= 0 or payload.get('output_voxels', 0) < 8:
        raise ValueError('Invalid reference output grid')
    return payload

"""Untrained CLF export using the authentic mCEL network; never calls fit()."""
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import yaml

from model.model_definition import get_model_from_yaml_definition
from data.mcel_data_provider import compute_grid_cell_origins_of_encoding

FEATURE_VERSION = 'clf-features-v4-wifi24'
MODEL_VERSION = 'clf-mcel-wifi24-v1'
SOURCE_COMMIT = 'ea17e54fdfad2cefe5c861f2b81e91a7a33c5688'


def build_clf_model(cell_count):
    if not isinstance(cell_count, int) or cell_count < 1:
        raise ValueError('cell_count must be a positive integer')
    path = Path(__file__).parents[1] / 'config/clf/wifi24.yml'
    conf = yaml.safe_load(path.read_text())
    return get_model_from_yaml_definition(conf, 24, [cell_count, cell_count * 2])


def make_feature_contract(ordered_bssids):
    bssids = [str(x).lower() for x in ordered_bssids]
    if (len(bssids) != 24 or len(set(bssids)) != 24 or
            any(not re.fullmatch(r'(?:[0-9a-f]{2}:){5}[0-9a-f]{2}', x) for x in bssids)):
        raise ValueError('provide exactly 24 unique canonical approved BSSIDs')
    return {
        'schema_version': '4.0', 'feature_version': FEATURE_VERSION,
        'model_version': MODEL_VERSION, 'model_inputs': ['input'],
        'wifi': {'input_name': 'input', 'dimensions': 24,
                 'clip_dbm': [-110, -20], 'missing_rssi_dbm': -110,
                 'normalization': '(clip(rssi_dbm, -110, -20) + 110) / 90',
                 'duplicate_policy': 'strongest_rssi', 'unknown_policy': 'ignore',
                 'access_points': [{'index': i, 'bssid': x} for i, x in enumerate(bssids)]},
        'motion': {'dimensions': 0, 'ordering': []}, 'ble': {'enabled': False},
    }


def make_grid_metadata(floors, grid_size_m, padding_ratio):
    size, padding = float(grid_size_m), float(padding_ratio)
    if not np.isfinite([size, padding]).all() or size <= 0 or not 0 <= padding < 1:
        raise ValueError('invalid grid size or padding')
    if sorted(f['floor'] for f in floors) != [0, 1]:
        raise ValueError('CLF requires explicit Ground=0 and First=1 dimensions')
    cells = []
    for floor in sorted(floors, key=lambda f: f['floor']):
        width, height = float(floor['width_m']), float(floor['height_m'])
        if not np.isfinite([width, height]).all() or min(width, height) <= 0:
            raise ValueError('floor extents must be positive calibrated metres')
        if (np.ceil(width / size) + 1) * (np.ceil(height / size) + 1) > 10000:
            raise ValueError('grid is too large; review cell size and extents')
        origins, _ = compute_grid_cell_origins_of_encoding(width, height, size)
        for local, (x, y) in enumerate(origins):
            cells.append({'index': len(cells), 'cell_id': f"F{floor['floor']}_C{local:04d}",
                          'floor': floor['floor'], 'origin_x_m': float(x), 'origin_y_m': float(y)})
    return {'feature_version': FEATURE_VERSION, 'model_version': MODEL_VERSION,
            'coordinate_units': 'metres', 'grid_size_m': size, 'padding_ratio': padding,
            'regression_scale_m': size / 2 + size * padding, 'cells': cells}


def write_untrained_package(model, directory, contract, grid, calibration):
    """Write a NEW directory. Returned package deliberately cannot localize live users.

    BSSIDs, metre dimensions and calibration are supplied by the caller; this
    function does not invent a CLF AP registry, geometry or measured calibration.
    """
    directory = Path(directory)
    if directory.exists():
        raise FileExistsError('export requires a new directory')
    if contract != make_feature_contract([x['bssid'] for x in contract['wifi']['access_points']]):
        raise ValueError('feature contract must match the Wi-Fi-only export specification')
    count = len(grid['cells'])
    if (model.input_shape != (None, 24) or
            [tuple(x) for x in model.output_shape] != [(None, count), (None, 2 * count)]):
        raise ValueError('network and registry shapes differ')
    directory.mkdir(parents=True)
    model.save(directory / 'clf_localization.keras')
    artifacts = {
        'feature_contract.json': contract,
        'access_points.json': {'feature_version': FEATURE_VERSION,
            'wifi': {x['bssid']: x['index'] for x in contract['wifi']['access_points']}},
        'preprocessing.json': {'feature_version': FEATURE_VERSION, 'wifi': {'method': 'clf_fixed_rssi'},
                               'motion': {'enabled': False}},
        'grid_metadata.json': grid, 'coordinate_calibration.json': calibration,
    }
    for filename, value in artifacts.items():
        (directory / filename).write_text(json.dumps(value, indent=2) + '\n')
    metadata = {
        'schema_version': '4.0', 'model_version': MODEL_VERSION, 'feature_version': FEATURE_VERSION,
        'architecture': 'mcel-mlp-v1', 'source_repository': 'GitMunish1/Clf-main',
        'source_commit': SOURCE_COMMIT,
        'builder_sha256': hashlib.sha256(Path(__file__).with_name('model_definition.py').read_bytes()).hexdigest(),
        'wifi_features': 24, 'motion_features': 0, 'ble_enabled': False,
        'model_input_names': ['input'], 'output_names': ['output_class', 'output_reg'],
        'decoder_min_confidence': 0.45, 'decoder_min_margin': 0.08,
        'status': 'untrained', 'training_metadata': None,
        'artifact_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()},
    }
    (directory / 'model_metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return directory

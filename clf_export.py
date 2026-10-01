"""Export a complete, version-checked CLF V1 Keras model package."""
from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np

from clf_features import FEATURE_VERSION, MOTION_DIM, WIFI_DIM, load_feature_contract
from data.mcel_data_provider import compute_grid_cell_origins_of_encoding


MODEL_VERSION = "clf-mcel-v1"


def _read_ready_calibration(path: Path) -> dict:
    calibration = json.loads(path.read_text(encoding="utf-8"))
    if calibration.get("calibrated") is not True or calibration.get("units") != "metres":
        raise ValueError("measured coordinate calibration is required before model export")
    transform = calibration.get("map_to_local_m") or {}
    definitions = transform.get("floors") or {"*": transform}
    if not definitions:
        raise ValueError("map_to_local_m calibration is required before model export")
    for definition in definitions.values():
        matrix = np.asarray(definition.get("matrix", []), dtype=float)
        offset = np.asarray(definition.get("offset", []), dtype=float)
        if matrix.shape != (2, 2) or offset.shape != (2,) or not np.isfinite(matrix).all() or not np.isfinite(offset).all():
            raise ValueError("valid map_to_local_m affine calibration is required before model export")
    return calibration


def build_grid_metadata(data_provider) -> dict:
    if not data_provider.grid_per_floor:
        raise ValueError("grid encoding must be generated before export")
    cells = []
    global_index = 0
    grid_size = float(data_provider.pr.get_param("grid_size"))
    padding_ratio = float(data_provider.pr.get_param("padding_ratio"))
    for floor_index, floor_value in enumerate(data_provider.floors):
        origins, count = compute_grid_cell_origins_of_encoding(
            data_provider.floorplan_width[floor_index],
            data_provider.floorplan_height[floor_index],
            grid_size,
        )
        for local_index in range(count):
            cells.append({
                "index": global_index,
                "cell_id": f"F{int(floor_value)}_C{local_index:04d}",
                "floor": int(floor_value),
                "origin_x_m": float(origins[local_index, 0]),
                "origin_y_m": float(origins[local_index, 1]),
            })
            global_index += 1
    return {
        "schema_version": "1.0",
        "feature_version": FEATURE_VERSION,
        "model_version": MODEL_VERSION,
        "coordinate_units": "metres",
        "grid_size_m": grid_size,
        "padding_ratio": padding_ratio,
        "regression_scale_m": grid_size / 2.0 + grid_size * padding_ratio,
        "cells": cells,
    }


def export_model_package(model, data_provider, output_dir, calibration_path, contract_path=None):
    """Save model plus every artifact required by backend inference.

    This function intentionally refuses partial packages. Templates in the
    repository remain marked not-ready until real survey training occurs.
    """
    output = Path(output_dir)
    calibration_path = Path(calibration_path)
    _read_ready_calibration(calibration_path)
    contract_path = Path(contract_path) if contract_path else None
    contract = load_feature_contract(contract_path)
    preprocessing = data_provider.preprocessing_artifact()
    grid = build_grid_metadata(data_provider)
    expected_inputs = [tensor.name.split(":")[0] for tensor in model.inputs]
    if expected_inputs != contract["model_inputs"]:
        raise ValueError(f"model input mismatch: expected {contract['model_inputs']}, got {expected_inputs}")

    output.mkdir(parents=True, exist_ok=True)
    model.save(output / "clf_localization.keras")
    source_contract = contract_path or Path(__file__).resolve().parent / "artifacts/contracts/feature_contract.json"
    shutil.copyfile(source_contract, output / "feature_contract.json")
    shutil.copyfile(calibration_path, output / "coordinate_calibration.json")
    (output / "preprocessing.json").write_text(json.dumps(preprocessing, indent=2) + "\n", encoding="utf-8")
    (output / "grid_metadata.json").write_text(json.dumps(grid, indent=2) + "\n", encoding="utf-8")
    access_points = {
        "schema_version": "2.0",
        "feature_version": FEATURE_VERSION,
        "wifi": {item["bssid"]: item["index"] for item in contract["wifi"]["access_points"]},
        "metadata": contract["wifi"]["access_points"],
    }
    (output / "access_points.json").write_text(json.dumps(access_points, indent=2) + "\n", encoding="utf-8")
    metadata = {
        "schema_version": "1.0",
        "model_version": MODEL_VERSION,
        "feature_version": FEATURE_VERSION,
        "wifi_features": WIFI_DIM,
        "motion_features": MOTION_DIM,
        "ble_enabled": False,
        "model_input_names": contract["model_inputs"],
        "output_names": ["output_class", "output_reg"],
        "decoder_min_confidence": float(data_provider._clf_param("decoder_min_confidence", 0.45)),
        "decoder_min_margin": float(data_provider._clf_param("decoder_min_margin", 0.08)),
        "threshold_status": "initial_requires_real_validation_tuning",
        "grid_size_m": grid["grid_size_m"],
        "grid_cells": len(grid["cells"]),
        "training_data": "real_survey_sessions",
    }
    (output / "model_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata

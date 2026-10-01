import json
from pathlib import Path

import numpy as np
import pytest

from clf_evaluation import ablation_inputs, localization_metrics
from clf_export import export_model_package
from clf_features import MotionScaler


class Params:
    values = {"grid_size": 3.0, "padding_ratio": 0.3}

    def get_param(self, name):
        return self.values[name]


class DummyProvider:
    grid_per_floor = [4]
    floors = np.asarray([0])
    floorplan_width = [3.0]
    floorplan_height = [3.0]
    pr = Params()
    motion_scaler = MotionScaler(np.zeros(19, dtype=np.float32), np.ones(19, dtype=np.float32))

    def preprocessing_artifact(self):
        return self.motion_scaler.to_artifact()

    @staticmethod
    def _clf_param(name, default):
        return {"decoder_min_confidence": 0.45, "decoder_min_margin": 0.08}.get(name, default)


class Tensor:
    def __init__(self, name):
        self.name = name


class DummyModel:
    inputs = [Tensor("wifi_input:0"), Tensor("motion_input:0")]

    @staticmethod
    def save(path):
        Path(path).write_text("test-model")


def test_complete_model_package_export(tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({
        "calibrated": True, "units": "metres",
        "map_to_local_m": {"matrix": [[1, 0], [0, 1]], "offset": [0, 0]},
    }))
    output = tmp_path / "package"
    metadata = export_model_package(DummyModel(), DummyProvider(), output, calibration)
    assert metadata["wifi_features"] == 13 and metadata["motion_features"] == 19
    assert metadata["ble_enabled"] is False
    assert {item.name for item in output.iterdir()} == {
        "clf_localization.keras", "feature_contract.json", "access_points.json",
        "preprocessing.json", "grid_metadata.json", "coordinate_calibration.json",
        "model_metadata.json",
    }
    grid = json.loads((output / "grid_metadata.json").read_text())
    assert [cell["cell_id"] for cell in grid["cells"]] == [
        "F0_C0000", "F0_C0001", "F0_C0002", "F0_C0003"
    ]


def test_export_rejects_uncalibrated_coordinates(tmp_path):
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"calibrated": False, "units": "metres"}))
    with pytest.raises(ValueError, match="coordinate calibration"):
        export_model_package(DummyModel(), DummyProvider(), tmp_path / "package", calibration)


def test_real_validation_metric_contract_and_motion_ablation():
    metrics = localization_metrics(
        true_floor=[0, 1], true_cell=[0, 2], true_xy_m=[[0, 0], [3, 4]],
        predicted_floor=[0, 1], predicted_cell=[0, 1], predicted_xy_m=[[0, 0], [0, 0]],
        accepted=[True, False], latency_ms=[2.0, 4.0],
    )
    assert metrics == {
        "floor_accuracy": 1.0, "grid_cell_accuracy": 0.5,
        "mean_position_error_m": 0.0, "median_position_error_m": 0.0,
        "p95_position_error_m": 0.0, "confidence_acceptance_rate": 0.5,
        "low_confidence_rejection_rate": 0.5, "mean_inference_latency_ms": 3.0,
    }
    inputs = {"wifi_input": np.ones((2, 13)), "motion_input": np.ones((2, 19))}
    variants = ablation_inputs(inputs)
    assert np.all(variants["wifi_only"]["motion_input"] == 0)
    assert np.all(variants["wifi_plus_motion"]["motion_input"] == 1)

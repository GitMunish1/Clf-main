import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import yaml

from clf_features import (
    MotionScaler,
    build_motion_vector,
    build_wifi_vector,
    load_feature_contract,
    normalize_bssid,
    normalize_rssi,
    ordered_bssids,
    preprocess_observation,
)
from data.clf_data_connector import CLFDataConnector
from data.clf_sensor_fusion_data_provider import CLFSensorFusionDataProvider
from data.dp_factory import _validate_clf_grid_scale


class CLFSensorFusionContractTest(unittest.TestCase):
    def setUp(self):
        self.contract = load_feature_contract()

    def _dataset(self, sessions=6):
        rows = []
        for i in range(30):
            session = f"survey-{i % sessions}"
            split = "train" if i % sessions < 4 else ("val" if i % sessions == 4 else "test")
            rows.append({
                "node_id": f"n{i % 5}", "x_m": float(i % 5), "y_m": float(i // 5),
                "floor": i % 2, "session_id": session, "split": split,
                "wifi_0C-7B-C8-E0-6C-C3": -42 - i,
                "wifi_e4:55:a8:58:99:c7": -80 + (i % 10),
                "wifi_aa:bb:cc:dd:ee:ff": -20,
                "accel_x": 0.1 * i, "accel_y": 0.2, "accel_z": 9.8,
                "gyro_x": 0.01, "gyro_y": 0.02 * i, "gyro_z": -0.01,
                "linear_accel_x": 0.03, "linear_accel_y": 0.04, "linear_accel_z": 0.05,
                "step_delta": i % 3, "heading_deg": 360 if i == 0 else i * 20,
            })
        return pd.DataFrame(rows)

    def _provider(self, path):
        connector = CLFDataConnector(csv_path=path, require_calibrated_coordinates=False)
        return CLFSensorFusionDataProvider({
            "val": 0.15, "grid_size": 3, "padding_ratio": 0.3,
            "max_grid_cells": 5000, "label_weight_epsilon": 1e-6,
            "decoder_top_k": 3, "decoder_secondary_ratio": 0.35,
            "decoder_neighbor_radius_cells": 1.5,
            "decoder_min_confidence": 0.45, "decoder_min_margin": 0.08,
        }, connector)

    def _temp_csv(self, frame):
        handle = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        handle.close()
        frame.to_csv(handle.name, index=False)
        self.addCleanup(lambda: Path(handle.name).unlink(missing_ok=True))
        return handle.name

    def test_fixed_ap_order_normalization_and_duplicates(self):
        self.assertEqual(len(ordered_bssids(self.contract)), 13)
        self.assertEqual(ordered_bssids(self.contract)[0], "0c:7b:c8:e0:6c:c3")
        self.assertEqual(ordered_bssids(self.contract)[-1], "e4:55:a8:39:d8:6c")
        self.assertEqual(normalize_bssid("0C-7B-C8-E0-6C-C3"), "0c:7b:c8:e0:6c:c3")
        self.assertAlmostEqual(normalize_rssi(-110), 0.0)
        self.assertAlmostEqual(normalize_rssi(-90), 20 / 90)
        self.assertAlmostEqual(normalize_rssi(-20), 1.0)
        vector = build_wifi_vector([
            {"bssid": "0c:7b:c8:e0:6c:c3", "rssi_dbm": -70},
            {"bssid": "0C-7B-C8-E0-6C-C3", "rssi_dbm": -42},
            {"bssid": "aa:bb:cc:dd:ee:ff", "rssi_dbm": -20},
        ], self.contract)
        self.assertAlmostEqual(vector[0], (-42 + 110) / 90, places=6)
        self.assertTrue(np.all(vector[1:] == 0.0))

    def test_named_inputs_are_exactly_13_plus_19_and_scaler_uses_train_only(self):
        path = self._temp_csv(self._dataset())
        provider = self._provider(path).load_dataset().generate_split_indices().generate_validation_indices()
        raw_motion = provider.rss[:, provider.dc.feature_groups["motion"]].copy()
        train = provider.split_indices[0]["train"]
        provider.build_sensor_inputs()
        self.assertEqual(set(provider.x), {"wifi_input", "motion_input"})
        self.assertEqual(provider.x["wifi_input"].shape, (30, 13))
        self.assertEqual(provider.x["motion_input"].shape, (30, 19))
        self.assertTrue(np.isfinite(provider.x["motion_input"]).all())
        np.testing.assert_allclose(provider.motion_scaler.mean[:17], np.mean(raw_motion[train, :17], axis=0))

    def test_motion_order_heading_wrap_and_missing_magnetometer(self):
        zero = build_motion_vector({"heading_deg": 0})
        wrapped = build_motion_vector({"heading_deg": 360})
        np.testing.assert_allclose(zero, wrapped, atol=1e-7)
        self.assertEqual(len(zero), 19)
        np.testing.assert_array_equal(zero[12:17], np.zeros(5))
        self.assertAlmostEqual(zero[17], 0.0)
        self.assertAlmostEqual(zero[18], 1.0)

    def test_golden_training_preprocessing(self):
        fixture = json.loads((Path(__file__).parent / "fixtures/clf_features_v2_golden.json").read_text())
        scaler = MotionScaler.from_artifact(fixture["preprocessing"])
        actual = preprocess_observation(fixture["observation"], self.contract, scaler)
        np.testing.assert_allclose(actual["wifi_input"], fixture["expected"]["wifi_input"], atol=1e-7)
        np.testing.assert_allclose(actual["motion_input"], fixture["expected"]["motion_input"], atol=1e-7)

    def test_session_split_never_leaks(self):
        frame = self._dataset(sessions=10).drop(columns="split")
        path = self._temp_csv(frame)
        connector = CLFDataConnector(csv_path=path, require_calibrated_coordinates=False).load_dataset()
        memberships = {}
        for partition, indices in connector.split_indices[0].items():
            for session in frame.iloc[indices]["session_id"].unique():
                self.assertNotIn(session, memberships)
                memberships[session] = partition
        self.assertEqual(set(memberships), set(frame["session_id"]))

    def test_coordinate_calibration_template_blocks_training(self):
        path = self._temp_csv(self._dataset())
        with self.assertRaisesRegex(ValueError, "coordinate calibration is not ready"):
            CLFDataConnector(csv_path=path).load_dataset()

    def test_no_nan_targets_and_confidence_rejection(self):
        path = self._temp_csv(self._dataset())
        provider = self._provider(path)
        provider = (provider.load_dataset().generate_split_indices().generate_validation_indices()
                    .build_sensor_inputs().transform_to_grid_encoding()
                    .compute_multilabel_aug_data(weighted_grid_labels=False))
        self.assertTrue(np.isfinite(provider.multi_grid_cell_labels).all())
        self.assertTrue(np.isfinite(provider.multi_labels).all())
        num_cells = int(provider.get_num_grid_cells())
        uncertain = np.full((1, num_cells), 1.0 / num_cells)
        self.assertFalse(bool(provider.get_prediction_confidence(uncertain)["accepted"][0]))

    def test_obviously_uncalibrated_scale_is_rejected(self):
        class Dummy:
            floorplan_width = [1100.0]
            floorplan_height = [700.0]
        with self.assertRaisesRegex(ValueError, "raw GeoJSON/map units"):
            _validate_clf_grid_scale(Dummy(), {"grid_size": 3, "max_grid_cells": 5000})

    def test_v1_architecture_config_has_two_inputs_and_no_ble_encoder(self):
        config = yaml.safe_load((Path(__file__).parents[1] / "config/clf/config.yml").read_text())
        model = config["models"][0]
        self.assertEqual(model["modalities"], ["wifi", "motion"])
        self.assertEqual(model["encoders"]["wifi"]["layers"], [256, 128])
        self.assertEqual(model["encoders"]["motion"]["layers"], [64, 64])
        self.assertEqual(model["fusion"]["layers"], [256, 256])
        self.assertNotIn("ble", model["encoders"])


if __name__ == "__main__":
    unittest.main()

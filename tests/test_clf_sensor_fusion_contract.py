import os
import tempfile
import unittest

import numpy as np
import pandas as pd

from data.clf_data_connector import CLFDataConnector
from data.clf_sensor_fusion_data_provider import CLFSensorFusionDataProvider


class CLFSensorFusionContractTest(unittest.TestCase):
    def _dataset(self):
        rows = []
        for i in range(20):
            rows.append({
                'x': float(i % 5),
                'y': float(i // 5),
                'floor': 0,
                'wifi_ap1': -45 - i,
                'ble_b1': -60 - i,
                'steps': i,
                'step_delta': 1 if i else 0,
                'heading_deg': (i * 20) % 360,
                'imu_accel_mean_x': 0.1 * i,
                'split': (
                    'train' if i < 12
                    else ('val' if i < 16 else 'test')
                ),
            })
        return pd.DataFrame(rows)

    def _provider(self, path):
        conn = CLFDataConnector(csv_path=path)
        return CLFSensorFusionDataProvider(
            {
                'val': 0.2,
                'grid_size': 3,
                'padding_ratio': 0.3,
                'label_weight_epsilon': 1e-6,
                'decoder_top_k': 3,
                'decoder_secondary_ratio': 0.35,
                'decoder_neighbor_radius_cells': 1.5,
                'decoder_min_confidence': 0.45,
                'decoder_min_margin': 0.08,
            },
            conn,
        )

    def test_named_inputs_and_heading_encoding(self):
        with tempfile.NamedTemporaryFile(
            suffix='.csv', delete=False
        ) as f:
            path = f.name

        try:
            self._dataset().to_csv(path, index=False)
            dp = self._provider(path)
            dp = (
                dp.load_dataset()
                .generate_split_indices()
                .generate_validation_indices()
                .build_sensor_inputs()
            )

            self.assertEqual(
                set(dp.x),
                {'wifi_input', 'ble_input', 'motion_input'}
            )
            self.assertEqual(dp.x['wifi_input'].shape[0], 20)
            self.assertEqual(dp.x['ble_input'].shape[0], 20)
            self.assertGreaterEqual(
                dp.x['motion_input'].shape[1], 6
            )
            self.assertEqual(
                len(dp.split_indices[0]['val']), 4
            )
            self.assertTrue(
                np.all(dp.x['wifi_input'] >= 0.0)
            )
            self.assertTrue(
                np.all(dp.x['wifi_input'] <= 1.0)
            )
        finally:
            os.unlink(path)

    def test_grid_origin_does_not_create_nan_targets(self):
        with tempfile.NamedTemporaryFile(
            suffix='.csv', delete=False
        ) as f:
            path = f.name

        try:
            # First sample is exactly at (0, 0), the old 1/distance path
            # could divide by zero for this case.
            self._dataset().to_csv(path, index=False)
            dp = self._provider(path)
            dp = (
                dp.load_dataset()
                .generate_split_indices()
                .generate_validation_indices()
                .build_sensor_inputs()
                .transform_to_grid_encoding()
                .compute_multilabel_aug_data(
                    weighted_grid_labels=False
                )
            )

            self.assertTrue(
                np.isfinite(dp.multi_grid_cell_labels).all()
            )
            self.assertTrue(
                np.isfinite(dp.multi_labels).all()
            )
            class_sums = np.sum(
                dp.multi_grid_cell_labels, axis=1
            )
            self.assertTrue(
                np.allclose(class_sums, 1.0)
            )
        finally:
            os.unlink(path)

    def test_low_confidence_prediction_is_rejected(self):
        with tempfile.NamedTemporaryFile(
            suffix='.csv', delete=False
        ) as f:
            path = f.name

        try:
            self._dataset().to_csv(path, index=False)
            dp = self._provider(path)
            dp = (
                dp.load_dataset()
                .generate_split_indices()
                .generate_validation_indices()
                .build_sensor_inputs()
                .transform_to_grid_encoding()
            )

            num_cells = int(dp.get_num_grid_cells())
            uncertain = np.full(
                (1, num_cells), 1.0 / num_cells
            )
            status = dp.get_prediction_confidence(
                uncertain
            )
            self.assertFalse(bool(status['accepted'][0]))

            confident = np.zeros((1, num_cells))
            confident[0, 0] = 0.9
            if num_cells > 1:
                confident[0, 1] = 0.1
            status = dp.get_prediction_confidence(
                confident
            )
            self.assertTrue(bool(status['accepted'][0]))
        finally:
            os.unlink(path)


if __name__ == '__main__':
    unittest.main()

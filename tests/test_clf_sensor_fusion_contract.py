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

    def test_named_inputs_and_heading_encoding(self):
        with tempfile.NamedTemporaryFile(
            suffix='.csv', delete=False
        ) as f:
            path = f.name

        try:
            self._dataset().to_csv(path, index=False)
            conn = CLFDataConnector(csv_path=path)
            dp = CLFSensorFusionDataProvider(
                {
                    'val': 0.2,
                    'grid_size': 3,
                    'padding_ratio': 0.3
                },
                conn
            )
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


if __name__ == '__main__':
    unittest.main()

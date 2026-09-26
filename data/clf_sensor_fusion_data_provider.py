import numpy as np

from data.mcel_data_provider import MCELdataProvider


class CLFSensorFusionDataProvider(MCELdataProvider):
    """Multi-input data provider for CLF's sensor-fusion Multi-CEL model.

    Keras receives three named inputs:
      - wifi_input: normalized Wi-Fi RSSI fingerprint
      - ble_input: normalized BLE RSSI fingerprint
      - motion_input: steps, heading sin/cos, distance and flattened IMU features

    The existing Multi-CEL labels/decoding are inherited unchanged.
    """

    def build_sensor_inputs(self):
        raw = self.rss.astype(np.float32, copy=True)
        groups = self.dc.feature_groups
        feature_names = self.dc.feature_names
        n = len(raw)

        wifi = self._slice_or_zero(raw, groups.get('wifi', []), n)
        ble = self._slice_or_zero(raw, groups.get('ble', []), n)
        motion_idx = groups.get('motion', [])
        motion = self._slice_or_zero(raw, motion_idx, n)

        wifi = self._normalize_rssi(wifi)
        ble = self._normalize_rssi(ble)
        motion = self._normalize_motion(
            motion, [feature_names[i] for i in motion_idx])

        self.x = {
            'wifi_input': wifi.astype(np.float32),
            'ble_input': ble.astype(np.float32),
            'motion_input': motion.astype(np.float32),
        }
        return self

    @staticmethod
    def _slice_or_zero(raw, indices, n):
        if not indices:
            return np.zeros((n, 1), dtype=np.float32)
        return raw[:, indices]

    @staticmethod
    def _normalize_rssi(values):
        # CLF uses -110 dBm for missing/very weak radios. Map the expected
        # RSSI range [-110, 0] to [0, 1].
        result = (values + 110.0) / 110.0
        return np.clip(result, 0.0, 1.0)

    def _normalize_motion(self, values, names):
        result = values.astype(np.float32, copy=True)
        if not names:
            return result

        # Fit scaling only on the training partition to avoid validation/test
        # leakage. Heading sin/cos are already in [-1, 1].
        train_idx = self.split_indices[self.split_idx]['train']
        for col, name in enumerate(names):
            if name in ('heading_sin', 'heading_cos'):
                continue
            train_values = (
                result[train_idx, col] if len(train_idx) else result[:, col]
            )
            scale = max(
                float(np.max(np.abs(train_values)))
                if len(train_values) else 0.0,
                1.0,
            )
            result[:, col] /= scale
        return result

    def get_x(self, partition='train'):
        subset = self.split_indices[self.split_idx][partition]
        return {name: values[subset] for name, values in self.x.items()}

    def get_input_dim(self):
        return {
            name: int(values.shape[1])
            for name, values in self.x.items()
        }

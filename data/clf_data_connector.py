"""CLF multimodal dataset connector.

Expected CSV: one row per labelled fingerprint. Required metadata columns:
``x``, ``y`` and ``floor``.

Radio columns:
- ``wifi_<stable-bssid-or-id>``
- ``ble_<stable-beacon-id>``

Motion columns can include ``steps``, ``step_delta``, ``heading_deg``,
``distance_m`` and any flattened ``imu_*`` values. Heading is converted to
sine/cosine before training so that 359 degrees remains close to 0 degrees.
Missing radio RSSI values are represented as -110 dBm.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from data.data_connector import DatasetConnector
from utils.definitions import get_project_root


class CLFDataConnector(DatasetConnector):
    def __init__(self, csv_path="datasets/clf/fingerprints.csv",
                 wifi_prefix="wifi_", ble_prefix="ble_", imu_prefix="imu_",
                 motion_features=None, missing_rssi=-110.0):
        super().__init__()
        self.csv_path = csv_path
        self.wifi_prefix = wifi_prefix
        self.ble_prefix = ble_prefix
        self.imu_prefix = imu_prefix
        self.motion_features = motion_features or [
            "steps", "step_delta", "heading_sin", "heading_cos", "distance_m"
        ]
        self.missing_rssi = missing_rssi
        self.feature_names = []
        self.feature_groups = {}

    def load_dataset(self):
        path = Path(get_project_root()) / self.csv_path
        data = pd.read_csv(path)
        required = {"x", "y", "floor"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError(
                "CLF dataset missing required columns: {}".format(sorted(missing))
            )

        wifi = sorted(c for c in data.columns if c.startswith(self.wifi_prefix))
        ble = sorted(c for c in data.columns if c.startswith(self.ble_prefix))
        imu = sorted(c for c in data.columns if c.startswith(self.imu_prefix))
        if not wifi and not ble:
            raise ValueError(
                "CLF dataset needs at least one wifi_* or ble_* RSSI column"
            )

        if "heading_deg" in data.columns:
            radians = np.deg2rad(
                pd.to_numeric(data["heading_deg"], errors="coerce").fillna(0.0)
            )
            data["heading_sin"] = np.sin(radians)
            data["heading_cos"] = np.cos(radians)

        for name in [
            "steps", "step_delta", "distance_m", "heading_sin", "heading_cos"
        ]:
            if name not in data.columns:
                data[name] = 0.0

        motion_base = [c for c in self.motion_features if c in data.columns]
        motion = motion_base + [c for c in imu if c not in motion_base]

        self.feature_names = wifi + ble + motion
        wifi_end = len(wifi)
        ble_end = wifi_end + len(ble)
        self.feature_groups = {
            "wifi": list(range(0, wifi_end)),
            "ble": list(range(wifi_end, ble_end)),
            "motion": list(range(ble_end, len(self.feature_names))),
        }

        sensor_cols = wifi + ble
        data[sensor_cols] = (
            data[sensor_cols]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(self.missing_rssi)
        )
        if motion:
            data[motion] = (
                data[motion].apply(pd.to_numeric, errors="coerce").fillna(0.0)
            )

        self.rss = data[self.feature_names].to_numpy(dtype=np.float32)
        self.pos = data[["x", "y"]].to_numpy(dtype=np.float32)
        self.floor = data["floor"].to_numpy()
        self.floors = np.sort(np.unique(self.floor))
        self.num_floors = len(self.floors)

        self.floorplan_width = []
        self.floorplan_height = []
        for floor in self.floors:
            p = self.pos[self.floor == floor]
            self.floorplan_width.append(float(np.max(p[:, 0]) + 1e-6))
            self.floorplan_height.append(float(np.max(p[:, 1]) + 1e-6))

        if "timestamp" in data.columns:
            self.time = data["timestamp"].to_numpy()

        if "split" in data.columns:
            split = data["split"].astype(str).str.lower().to_numpy()
            self.split_indices = [{
                "train": np.where(split == "train")[0],
                "val": np.where(split == "val")[0],
                "test": np.where(split == "test")[0],
            }]
        return self

    def get_dataset_identifier(self):
        return "clf"

"""CLF multimodal dataset connector.

Expected CSV: one row per labelled fingerprint. Required metadata columns:
x, y, floor. Optional: timestamp, steps, heading_deg, distance_m.
Sensor columns use wifi_<id> and ble_<id>. Missing RSSI values are filled with -110 dBm.

The connector deliberately builds a stable feature order:
[all Wi-Fi RSSI] + [all BLE RSSI] + [motion features].
"""
from pathlib import Path
import numpy as np
import pandas as pd

from data.data_connector import DatasetConnector
from utils.definitions import get_project_root


class CLFDataConnector(DatasetConnector):
    def __init__(self, csv_path="datasets/clf/fingerprints.csv",
                 wifi_prefix="wifi_", ble_prefix="ble_",
                 motion_features=None, missing_rssi=-110.0):
        super().__init__()
        self.csv_path = csv_path
        self.wifi_prefix = wifi_prefix
        self.ble_prefix = ble_prefix
        self.motion_features = motion_features or ["steps", "heading_sin", "heading_cos", "distance_m"]
        self.missing_rssi = missing_rssi
        self.feature_names = []
        self.feature_groups = {}

    def load_dataset(self):
        path = Path(get_project_root()) / self.csv_path
        data = pd.read_csv(path)
        required = {"x", "y", "floor"}
        missing = required.difference(data.columns)
        if missing:
            raise ValueError("CLF dataset missing required columns: {}".format(sorted(missing)))

        wifi = sorted([c for c in data.columns if c.startswith(self.wifi_prefix)])
        ble = sorted([c for c in data.columns if c.startswith(self.ble_prefix)])
        if not wifi and not ble:
            raise ValueError("CLF dataset needs at least one wifi_* or ble_* RSSI column")

        # Encode heading cyclically so 359 degrees is close to 0 degrees.
        if "heading_deg" in data.columns:
            radians = np.deg2rad(pd.to_numeric(data["heading_deg"], errors="coerce").fillna(0.0))
            data["heading_sin"] = np.sin(radians)
            data["heading_cos"] = np.cos(radians)

        for name in ["steps", "distance_m", "heading_sin", "heading_cos"]:
            if name not in data.columns:
                data[name] = 0.0

        motion = [c for c in self.motion_features if c in data.columns]
        self.feature_names = wifi + ble + motion
        self.feature_groups = {
            "wifi": list(range(0, len(wifi))),
            "ble": list(range(len(wifi), len(wifi) + len(ble))),
            "motion": list(range(len(wifi) + len(ble), len(self.feature_names))),
        }

        sensor_cols = wifi + ble
        data[sensor_cols] = data[sensor_cols].apply(pd.to_numeric, errors="coerce").fillna(self.missing_rssi)
        data[motion] = data[motion].apply(pd.to_numeric, errors="coerce").fillna(0.0)

        self.rss = data[self.feature_names].to_numpy(dtype=np.float32)
        self.pos = data[["x", "y"]].to_numpy(dtype=np.float32)
        self.floor = data["floor"].to_numpy()
        self.floors = np.sort(np.unique(self.floor))
        self.num_floors = len(self.floors)

        # Multi-CEL grid encoding expects per-floor width/height in the same coordinate system.
        self.floorplan_width = []
        self.floorplan_height = []
        for floor in self.floors:
            p = self.pos[self.floor == floor]
            self.floorplan_width.append(float(np.max(p[:, 0]) + 1e-6))
            self.floorplan_height.append(float(np.max(p[:, 1]) + 1e-6))

        if "timestamp" in data.columns:
            self.time = data["timestamp"].to_numpy()

        # Optional explicit split column prevents leakage between collection walks.
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
